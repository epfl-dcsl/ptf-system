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
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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


#include <vector>
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "egress_gate.h"

namespace tensorflow {
    using namespace std;
    using namespace errors;

    EgressGate::EgressGate(const DataTypeVector &component_dtypes, const std::vector<TensorShape> &component_shapes,
                               const string &name, const int32 capacity, const bool limit_upstream) :
            StreamingGate(component_dtypes, component_shapes, name, capacity, limit_upstream, false) {}

  void EgressGate::TryDequeueForRequest(const IDandCountType &request_id, OpKernelContext *ctx,
                                        CallbackWithPlainResult callback) {
      auto status = claim_id(request_id);
      if (!status.ok()) {
          ctx->SetStatus(status);
          callback(Tuple());
      } else {
          TryDequeueCommon(ctx, [callback](const Tensor &id_and_count, const Tuple &data) { callback(Tuple()); }, kint32max, "DequeueDataset", [this, request_id, callback](Attempt *attempt) {
              auto *const context = attempt->context;
              if (closed_) {
                  context->SetStatus(Cancelled(name_, " is closed."));
                  return RunResult::kComplete;
              }

              auto open_request_p = FindOpenRequest(request_id);
              if (open_request_p != nullptr) {
                  auto &open_request = *open_request_p;
                  if (open_request.all_rounds_enqueued()) {
                      DCHECK_EQ(open_request.dequeued(), 0);
                      Tuple tuple;
                      auto status = open_request.DequeueMany(context, tuple, kint32max);
                      if (!status.ok()) {
                          context->SetStatus(status);
                      } else {
                          attempt->done_callback = [callback, tuple]() { callback(tuple); };
                          CloseRequest(open_request_p);
                          release_id(request_id);
                      }
                      return RunResult::kComplete;
                  }
              } else if (limit_requests_from_upstream_gate_ and num_opened_requests() == max_open_requests_) {
                  LOG(WARNING) << name_ << " blocked credit request set for id " << request_id << ". Need another credit request to clear the pipeline for this one.";
              } else if (exhausted()) {
                  context->SetStatus(OutOfRange(name_, " has no more elements."));
                  return RunResult::kComplete;
              }
              return RunResult::kNoProgress;
          });
      }
    }

  void EgressGate::TryDequeue(OpKernelContext *ctx, CallbackWithResult callback) {
    ctx->SetStatus(Unimplemented("EgressGate doesn't implement TryDequeue"));
    callback(Tensor(), Tuple());
  }

  void EgressGate::TryDequeueMany(int32 num_elements, OpKernelContext *ctx,
                                         GateInterface::CallbackWithResult callback) {
    ctx->SetStatus(Unimplemented("EgressGate doesn't implement TryDequeueMany"));
    callback(Tensor(), Tuple());
  }

    Status EgressGate::SupplyDownstreamCredits(const int32 credits) {
      return Unimplemented("Egress gate does not supply downstream credits");
    }

    Status EgressGate::claim_id(const GateInterface::IDandCountType &idc) {
        mutex_lock l(awaited_requests_lock_);
        if (awaited_requests_.count(idc) != 0) {
            return errors::Internal("Multiple DequeueDataset calls for id ", idc);
        }
        awaited_requests_.emplace(idc);
        return Status::OK();
    }

    void EgressGate::release_id(const GateInterface::IDandCountType &idc) {
        mutex_lock l(awaited_requests_lock_);
        auto num_erased = awaited_requests_.erase(idc);
        DCHECK_EQ(num_erased, 1);
    }

    Status EgressGate::MatchesNodeDef(const NodeDef &node_def) const {
        return MatchesNodeDefOpAndAttributes(node_def, "EgressGate");
    }

    void EgressGate::TryDequeuePartition(int32 num_elements, OpKernelContext *ctx,
                                         GateInterface::CallbackWithResult callback) {
        ctx->SetStatus(Unimplemented("Egress gate doesn't implement TryDequeuePartition"));
        callback(Tensor(), Tuple());
    }

}  // namespace tensorflow
