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
#pragma once

#include "streaming_gate.h"

namespace tensorflow {
  class IngressGate : public StreamingGate {
  public:
      using CallbackWithId = std::function<void(const IDandCountType id)>;

    IngressGate(const DataTypeVector &component_dtypes, const std::vector<TensorShape> &component_shapes,
                    const string &name, IDandCountType id, const int32 capacity, const bool limit_upstream,
                    const bool limit_downstream);

    void
    TryEnqueue(const Tensor &id_and_count, const Tuple &tuple, OpKernelContext *ctx, DoneCallback callback) override;

      void TryEnqueueMany(const Tensor &id_and_count, const Tuple &tuple, OpKernelContext *ctx,
                          DoneCallback callback) override;

      void
    TryEnqueueRequest(const Tuple &dataset, OpKernelContext *ctx, CallbackWithId callback);

      Status MatchesNodeDef(const NodeDef &node_def) const override;

  private:
      IDandCountType id_ = 0;
      TF_DISALLOW_COPY_AND_ASSIGN(IngressGate);
  };
} // namespace tensorflow {
