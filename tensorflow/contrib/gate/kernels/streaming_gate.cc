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
#include "tensorflow/contrib/gate/framework/streaming_gate.h"
#include "tensorflow/contrib/gate/framework/gate_op.h"

namespace tensorflow {
  using namespace std;

  namespace {
    const string op_name("StreamingGate");
  }

  class StreamingGateOp : public GateOp<GateInterface> {
  public:
    explicit StreamingGateOp(OpKernelConstruction *ctx) : GateOp(ctx) {}

  private:
    Status CreateResource(GateInterface **resource) override {
      *resource = new StreamingGate(component_types_, component_shapes_, name(), capacity_, limit_upstream_,
                                    limit_downstream_);
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), StreamingGateOp);
} // namespace tensorflow {
