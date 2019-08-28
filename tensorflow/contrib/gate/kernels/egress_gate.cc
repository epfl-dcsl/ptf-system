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
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/contrib/gate/framework/egress_gate.h"

namespace tensorflow {

  using namespace std;

  namespace {
    const string op_name("EgressGate");
  }

  class EgressGateOp : public ResourceOpKernel<EgressGate> {
  public:
      explicit EgressGateOp(OpKernelConstruction* context) : ResourceOpKernel<EgressGate>(context) {
          OP_REQUIRES_OK(context,
                         context->GetAttr("component_types", &component_types_));
          OP_REQUIRES_OK(context,
                         context->GetAttr("component_shapes", &component_shapes_));
          OP_REQUIRES_OK(context,
                         context->GetAttr("limit_upstream", &limit_upstream_));
          OP_REQUIRES_OK(context,
                         context->GetAttr("capacity", &capacity_));
      }

  private:
    Status CreateResource(EgressGate **resource) override {
      *resource = new EgressGate(component_types_, component_shapes_, name(), capacity_, limit_upstream_);
      return Status::OK();
    }

      Status VerifyResource(EgressGate* barrier) override {
          return barrier->MatchesNodeDef(def());
      }

      DataTypeVector component_types_;
      std::vector<TensorShape> component_shapes_;
      bool limit_upstream_;
      int capacity_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), EgressGateOp);
} // namespace tensorflow {
