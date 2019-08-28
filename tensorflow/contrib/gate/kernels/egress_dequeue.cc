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
#include "tensorflow/contrib/gate/framework/gate_access_op.h"
#include "tensorflow/contrib/gate/framework/egress_gate.h"

namespace tensorflow {

  using namespace std;

  namespace {
    const string op_name("EgressDequeue");
  }

  class GateEgressDequeueOp : public GateAccessOp<EgressGate> {
  public:
    explicit GateEgressDequeueOp(OpKernelConstruction *ctx) : GateAccessOp<EgressGate>(ctx) {
    }

  protected:
    void ComputeWithGate(OpKernelContext *ctx, EgressGate *barrier, DoneCallback done) override {
        const Tensor *dataset_id_t;
        OP_REQUIRES_OK(ctx, ctx->input("requested_dataset_id", &dataset_id_t));
        auto dataset_id = dataset_id_t->scalar<int32>()();
        barrier->TryDequeueForRequest(dataset_id, ctx, [ctx, done](const EgressGate::Tuple &tuple) {
            if (ctx->status().ok()) {
                OpOutputList output_components;
                OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("components", &output_components), done);
                for (uint_fast32_t i = 0; i < tuple.size(); ++i) {
                    auto &t = tuple[i];
                    auto dtype = t.dtype();
                    auto shape = t.shape();
                    output_components.set(i, tuple[i]);
                }
            }
            done();
        });
    }
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), GateEgressDequeueOp);
} // namespace tensorflow {
