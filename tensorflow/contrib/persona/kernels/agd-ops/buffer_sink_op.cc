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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  class BufferSinkOp : public OpKernel {
  public:
    BufferSinkOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      ResourceContainer<Data> *buf;
      const Tensor *input;
      OP_REQUIRES_OK(ctx, ctx->input("data", &input));
      auto data = input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &buf));
      core::ScopedUnref a(buf);
      {
        ResourceReleaser<Data> b(*buf); // make sure destructs first
      }
    }
  };


REGISTER_KERNEL_BUILDER(Name("BufferSink").Device(DEVICE_CPU), BufferSinkOp);
} // namespace tensorflow {
