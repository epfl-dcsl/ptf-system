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
  class GateSupplyCreditsOp : public GateAccessOp<GateType> {
  public:
    explicit GateSupplyCreditsOp(OpKernelConstruction *ctx) : GateAccessOp<GateType>(ctx) {
        static_assert(is_base_of<GateInterface,GateType>::value, "Not derived from GateInterface");
    }

  protected:
    void ComputeWithGate(OpKernelContext *ctx, GateType *gate, GateInterface::DoneCallback done) override {
        const Tensor *credits_t;
        OP_REQUIRES_OK_ASYNC(ctx, ctx->input("credits", &credits_t), done);
        auto num_credits = credits_t->scalar<int32>()();
        // Note: the gate has a check for the number of credits. no need to do it here
        OP_REQUIRES_OK_ASYNC(ctx, gate->SupplyDownstreamCredits(num_credits), done);
        done();
    }
  };

} // namespace tensorflow {
