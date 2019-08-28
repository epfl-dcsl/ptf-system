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

#include <deque>

#include "gate_interface.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {

// Defines a GateOp, an abstract class for Barrier construction ops.
  template <typename GateType>
class GateOp : public ResourceOpKernel<GateType> {
 public:
  explicit GateOp(OpKernelConstruction* context) : ResourceOpKernel<GateType>(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("component_types", &component_types_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("component_shapes", &component_shapes_));
      OP_REQUIRES_OK(context,
                     context->GetAttr("capacity", &capacity_));
      OP_REQUIRES_OK(context,
                     context->GetAttr("limit_upstream", &limit_upstream_));
      OP_REQUIRES_OK(context,
                     context->GetAttr("limit_downstream", &limit_downstream_));
  }

 protected:
  // Variables accessible by subclasses
  DataTypeVector component_types_;
  std::vector<TensorShape> component_shapes_;
    int capacity_;
    bool limit_upstream_, limit_downstream_;

 private:
  using ResourceOpKernel<GateType>::def;
  Status VerifyResource(GateType* barrier) override {
    return barrier->MatchesNodeDef(def());
  }
};

}  // namespace tensorflow
