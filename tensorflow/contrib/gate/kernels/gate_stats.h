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

namespace tensorflow {
    using namespace std;

    namespace {
        inline Status
        assign_value_to_scalar(OpKernelContext *ctx, const string &name, size_t const value) {
            Tensor *value_t;
            TF_RETURN_IF_ERROR(ctx->allocate_output(name, {}, &value_t));
            value_t->scalar<int32>()() = value;
            return Status::OK();
        }

    }

  template <typename GateType>
  class NumOpenRequestsOp : public SynchronousGateAccessOp<GateType> {
  public:
    explicit NumOpenRequestsOp(OpKernelConstruction *ctx) : SynchronousGateAccessOp<GateType>(ctx) {
      static_assert(is_base_of<GateInterface,GateType>::value, "Not derived from GateInterface");
    }

  protected:
    void ComputeWithGate(OpKernelContext *ctx, GateType *gate) override {
        OP_REQUIRES_OK(ctx, assign_value_to_scalar(ctx, "open_requests", gate->OpenRequestCount()));
    }
  };

    template <typename GateType>
    class NumRoundsOp : public SynchronousGateAccessOp<GateType> {
    public:
        explicit NumRoundsOp(OpKernelConstruction *ctx) : SynchronousGateAccessOp<GateType>(ctx) {
            static_assert(is_base_of<GateInterface,GateType>::value, "Not derived from GateInterface");
        }

    protected:
        void ComputeWithGate(OpKernelContext *ctx, GateType *gate) override {
            OP_REQUIRES_OK(ctx, assign_value_to_scalar(ctx, "num_rounds", gate->NumOpenRounds()));
        }
    };

} // namespace tensorflow {
