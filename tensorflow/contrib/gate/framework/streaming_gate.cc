/* Copyright 2019 École Polytechnique Fédérale de Lausanne. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "streaming_gate.h"

#include <numeric>

namespace tensorflow {
  using namespace std;
    using namespace errors;

  StreamingGate::StreamingGate(const DataTypeVector &component_dtypes,
                                 const std::vector<TensorShape> &component_shapes, const string &name,
                                 const int32 capacity, const bool limit_upstream,
                                 const bool limit_downstream) :
          GateBase(component_dtypes, component_shapes, name, capacity, limit_upstream, limit_downstream) { }

  void StreamingGate::TryEnqueue(const Tensor &id_and_count, const Tuple &tuple, OpKernelContext *ctx, DoneCallback callback) {
      TryEnqueueDefault(ctx, callback, 1, "Enqueue", id_and_count, [this, tuple](Attempt *attempt, OpenRequest &open_request) {
          auto *const context = attempt->context;
          auto status = open_request.Enqueue(context, tuple);
          return status;
      });
  }

    void StreamingGate::TryEnqueueMany(const Tensor &id_and_count, const GateInterface::Tuple &tuple, OpKernelContext *ctx,
                                  GateInterface::DoneCallback callback) {
        const auto batch_size = tuple[0].dim_size(0);
        if (batch_size == 0) {
            ctx->SetStatus(Internal("Got an empty batch for StreamingGate::TryEnqueueMany"));
            return;
        }
        TryEnqueueDefault(ctx, callback, batch_size, "EnqueueMany", id_and_count, [this, tuple](Attempt *attempt, OpenRequest &open_request) {
            auto *const context = attempt->context;
            auto status = open_request.EnqueueBatch(context, tuple);
            return status;
        });
    }

  void StreamingGate::TryDequeue(OpKernelContext *ctx, CallbackWithResult callback) {
      TryDequeueDefault(ctx, 1, "Dequeue", callback,
                        [this, callback](Attempt *attempt, OpenRequest &open_req) {
                            auto * const context = attempt->context;
                            Tuple tuple;
                            TF_RETURN_IF_ERROR(open_req.Dequeue(context, tuple));
                            Tensor idc(*open_req.id_and_count(context));
                            attempt->done_callback = [callback, tuple, idc]() { callback(idc, tuple); };
                            return Status::OK();
                        });
  }

  void StreamingGate::TryDequeueMany(int32 num_elements, OpKernelContext *ctx, CallbackWithResult callback) {
      DCHECK_GT(num_elements, 0);
      TryDequeueDefault(ctx, num_elements, "DequeueMany", callback,
                        [this, callback](Attempt *attempt, OpenRequest &open_req) {
                            auto * const context = attempt->context;
                            Tuple tuple;
                            if (open_req.dequeued() == 0) {
                                open_req.set_id_and_count_batching(context, attempt->elements_requested);
                            }
                            // Need this min in case it is closing
                            TF_RETURN_IF_ERROR(open_req.DequeueMany(context, tuple, attempt->elements_requested));
                            Tensor idc(*open_req.id_and_count(context));
                            attempt->done_callback = [callback, tuple, idc]() { callback(idc, tuple); };
                            return Status::OK();
                        });
  }

  Status StreamingGate::MatchesNodeDef(const NodeDef &node_def) const {
      return MatchesNodeDefOpAndAttributes(node_def, "StreamingGate");
  }

    void StreamingGate::TryDequeuePartition(int32 num_elements, OpKernelContext *ctx,
                                            GateInterface::CallbackWithResult callback) {
        DCHECK_GT(num_elements, 0);
        TryDequeueDefault(ctx, num_elements, "DequeuePartition", callback, [this, callback](Attempt *attempt, OpenRequest &open_request) {
            auto * const context = attempt->context;
            Tuple tuple;

            TF_RETURN_IF_ERROR(open_request.DequeueMany(context, tuple, attempt->elements_requested));
            auto elements_received = tuple.at(0).dim_size(0);
            DCHECK_LE(elements_received, attempt->elements_requested);
            DCHECK_GT(elements_received, 0);

            Tensor id_and_count;
            auto idc = open_request.id_and_count(context);
            TF_RETURN_IF_ERROR(allocate_new_id_and_count(context, idc, elements_received, id_and_count));
            attempt->done_callback = [callback, tuple, id_and_count]() { callback(id_and_count, tuple); };
            return Status::OK();
        });
    }

    Status StreamingGate::allocate_new_id_and_count(OpKernelContext *ctx, const Tensor *id_and_count_parent,
                                                    const int32 num_elements, Tensor &out_tensor) {
        auto parent_dim_size = id_and_count_parent->dim_size(0);
        auto element_dim_size = id_and_count_parent->dim_size(1);
        DCHECK_EQ(element_dim_size, 2);
        TF_RETURN_IF_ERROR(ctx->allocate_temp(id_and_count_parent->dtype(),
                                              {parent_dim_size+1, element_dim_size},
                                              &out_tensor));
        auto out_as_matrix = out_tensor.matrix<int32>();
        auto parent_as_matrix = id_and_count_parent->matrix<int32>();
        for (size_t i = 0; i < parent_dim_size; ++i) {
            out_as_matrix(i, 0) = parent_as_matrix(i, 0);
            out_as_matrix(i, 1) = parent_as_matrix(i, 1);
        }
        out_as_matrix(parent_dim_size, 0) = id_++;
        out_as_matrix(parent_dim_size, 1) = num_elements;
        return Status::OK();
    }

} // namespace tensorflow {
