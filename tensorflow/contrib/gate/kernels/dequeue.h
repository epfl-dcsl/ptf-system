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

  template <typename GateType>
  class GateDequeueOp : public GateAccessOp<GateType> {
  public:
    explicit GateDequeueOp(OpKernelConstruction *ctx) : GateAccessOp<GateType>(ctx) {
        static_assert(is_base_of<GateInterface,GateType>::value, "Not derived from GateInterface");
    }

  protected:
    void ComputeWithGate(OpKernelContext *ctx, GateType *barrier, GateInterface::DoneCallback done) override {
      barrier->TryDequeue(ctx, [ctx, done](const Tensor &id_and_count, const typename GateType::Tuple &tuple) {
          if (ctx->status().ok()) {
            OpOutputList output_components;
            OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("components", &output_components), done);
            for (uint_fast32_t i = 0; i < tuple.size(); ++i) {
              auto &t = tuple[i];
              auto dtype = t.dtype();
              auto shape = t.shape();
              output_components.set(i, tuple[i]);
            }
            OP_REQUIRES_OK_ASYNC(ctx, ctx->set_output("id_and_count", id_and_count), done);
          }
          done();
      });
    }
  };
} // namespace tensorflow {
