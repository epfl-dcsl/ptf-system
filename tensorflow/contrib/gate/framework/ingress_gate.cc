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
#include "ingress_gate.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  void IngressGate::TryEnqueue(const Tensor &id_and_count, const GateInterface::Tuple &tuple,
                                      OpKernelContext *ctx, GateInterface::DoneCallback callback) {
    ctx->SetStatus(Unimplemented("Ingress Gate does not allow normal enqueuing"));
    callback();
  }

  IngressGate::IngressGate(const DataTypeVector &component_dtypes, const std::vector<TensorShape> &component_shapes,
                             const string &name, IDandCountType id, const int32 capacity, const bool limit_upstream,
                             const bool limit_downstream)
          : StreamingGate(component_dtypes, component_shapes, name, capacity, limit_upstream, limit_downstream), id_(id) { }

  void IngressGate::TryEnqueueRequest(const GateInterface::Tuple &dataset, OpKernelContext *ctx,
                                      CallbackWithId callback) {
      auto num_elements = dataset.at(0).dim_size(0);
      DCHECK_GT(num_elements, 0);
      Tensor id_and_count;
      auto status = ctx->allocate_temp(DT_INT32, {1, 2}, &id_and_count);
      if (!status.ok()) {
          ctx->SetStatus(status);
          callback(0); return;
      }
      auto id_and_count_matrix = id_and_count.matrix<int32>();
      auto assigned_id = id_++;
      id_and_count_matrix(0, 0) = assigned_id;
      id_and_count_matrix(0, 1) = num_elements;
      TryEnqueueCommon(ctx, [callback]() { callback(0); }, num_elements, "EnqueueDataset", [this, assigned_id, id_and_count, callback, dataset](Attempt *attempt) {
          auto *const context = attempt->context;
          if (closed_) {
              context->SetStatus(Cancelled(name_, " is closed."));
          } else {
              DCHECK_EQ(FindOpenRequest(assigned_id), nullptr);
              if (not limit_requests_to_downstream_gate_ or num_opened_requests() < max_open_requests_) {
                  auto status_pair = FindOrMakeNewRequest(id_and_count);
                  if (not status_pair.first.ok()) {
                      context->SetStatus(status_pair.first);
                  } else {
                      auto open_request_p = status_pair.second;
                      auto status = open_request_p->EnqueueBatch(context, dataset);
                      if (not status.ok()) {
                          context->SetStatus(status);
                      } else {
                          attempt->done_callback = [assigned_id, callback]() { callback(assigned_id); };
                      }
                  } // will fall through to kComplete
              } else {
                  return RunResult::kProgressImpossible;
              }
          }
          return RunResult::kComplete;
      });
  }

    Status IngressGate::MatchesNodeDef(const NodeDef &node_def) const {
        return GateBase::MatchesNodeDefOpAndAttributes(node_def, "IngressGate");
    }

    void
    IngressGate::TryEnqueueMany(const Tensor &id_and_count, const GateInterface::Tuple &tuple, OpKernelContext *ctx,
                                GateInterface::DoneCallback callback) {
        ctx->SetStatus(Unimplemented("Ingress Gate does not allow normal enqueuing"));
        callback();
    }

} // namespace tensorflow {
