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
#include <type_traits>
#include "tensorflow/contrib/gate/framework/gate_access_op.h"
#include "tensorflow/contrib/gate/framework/gate_interface.h"

namespace tensorflow {
    using namespace std;

  template <typename GateType>
  class GateEnqueueOp : public GateAccessOp<GateType> {
  public:
    explicit GateEnqueueOp(OpKernelConstruction *ctx) : GateAccessOp<GateType>(ctx) {
        static_assert(is_base_of<GateInterface,GateType>::value, "Not derived from GateInterface");
    }

  protected:
    void ComputeWithGate(OpKernelContext *ctx, GateType *barrier, GateInterface::DoneCallback done) override {
        const Tensor *id_and_count_t;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->input("id_and_count", &id_and_count_t), done);

        OpInputList components;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("components", &components), done);
        auto &dtypes = barrier->component_dtypes();

        OP_REQUIRES_ASYNC(ctx, dtypes.size() == components.size(), errors::Internal("input components doesn't match barrier size for Enqueue Op"), done);
        GateInterface::Tuple t;
        for (const auto &component : components) {
            t.push_back(component);
        }

        OP_REQUIRES_OK_ASYNC(ctx, barrier->ValidateTuple(t), done);

        barrier->TryEnqueue(*id_and_count_t, t, ctx, done);
    }

  private:
  };

} // namespace tensorflow {
