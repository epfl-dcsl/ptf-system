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
#include "tensorflow/contrib/gate/framework/ingress_gate.h"

namespace tensorflow {

  using namespace std;

  namespace {
    const string op_name("IngressEnqueue");
  }

  class GateIngressEnqueueOp : public GateAccessOp<IngressGate> {
  public:
    explicit GateIngressEnqueueOp(OpKernelConstruction *ctx) : GateAccessOp<IngressGate>(ctx) {
    }

  protected:
    void ComputeWithGate(OpKernelContext *ctx, IngressGate *barrier, DoneCallback done) override {
      IngressGate::Tuple tuple;
      OpInputList components;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("components", &components), done);
      for (const auto& component : components) {
        tuple.push_back(component);
      }
      OP_REQUIRES_OK_ASYNC(ctx, barrier->ValidateManyTuple(tuple), done);
        barrier->TryEnqueueRequest(tuple, ctx, [this, ctx, done](const IngressGate::IDandCountType id) {
            if (ctx->status().ok()) {
                Tensor *id_out_t;
                OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output("id_and_count", {}, &id_out_t), done);
                id_out_t->scalar<int32>()() = id;
            }
            done();
        });
    }
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), GateIngressEnqueueOp);
} // namespace tensorflow {
