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
  class GateDequeuePartitionOp : public GateAccessOp<GateType> {
  public:
    explicit GateDequeuePartitionOp(OpKernelConstruction *ctx) : GateAccessOp<GateType>(ctx) {
        static_assert(is_base_of<GateInterface,GateType>::value, "Not derived from GateInterface");
      OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    }

  protected:
    void ComputeWithGate(OpKernelContext *ctx, GateType *barrier, GateInterface::DoneCallback done) override {
      barrier->TryDequeuePartition(batch_size_, ctx, [ctx, done](const Tensor &id_and_count, const typename GateType::Tuple &tuple) {
        if (!ctx->status().ok()) {
          done();
          return;
        }
        OpOutputList output_components;
        OP_REQUIRES_OK_ASYNC( ctx, ctx->output_list("components", &output_components), done);
        for (uint_fast32_t i = 0; i < tuple.size(); ++i) {
          output_components.set(i, tuple[i]);
        }
        OP_REQUIRES_OK_ASYNC(ctx, ctx->set_output("id_and_count", id_and_count), done);
        done();
      });
    }

  private:
    int32 batch_size_;
  };

} // namespace tensorflow {
