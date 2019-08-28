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
    using namespace errors;

  template <typename GateType>
  class GateRequestCreditsOp : public GateAccessOp<GateType> {
  public:
    explicit GateRequestCreditsOp(OpKernelConstruction *ctx) : GateAccessOp<GateType>(ctx) {
        static_assert(is_base_of<GateInterface,GateType>::value, "Not derived from GateInterface");
    }

  protected:
    void ComputeWithGate(OpKernelContext *ctx, GateType *gate, GateInterface::DoneCallback done) override {
        gate->TryRequestUpstreamCredits(ctx, [ctx, done](int32 available_credits) {
            OP_REQUIRES_ASYNC(ctx, available_credits > 0, Internal("Got a non-positive number of credits: ", available_credits), done);
            if (ctx->status().ok()) {
                Tensor *credits_out_t;
                OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output("credits", {}, &credits_out_t), done);
                credits_out_t->scalar<int32>()() = available_credits;
            }
           done();
        });
    }
  };

} // namespace tensorflow {
