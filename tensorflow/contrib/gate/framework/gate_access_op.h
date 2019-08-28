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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/gate/framework/gate_interface.h"

namespace tensorflow {

  template <typename GateType>
  class GateAccessOp : public AsyncOpKernel {
  public:
    using DoneCallback = AsyncOpKernel::DoneCallback;

    explicit GateAccessOp(OpKernelConstruction *ctx) : AsyncOpKernel(ctx) {}

    void ComputeAsync(OpKernelContext *ctx, DoneCallback done) final {
        GateType *gate;
        OP_REQUIRES_OK_ASYNC(ctx, GetResourceFromContext(ctx, "handle", &gate), done);
        core::ScopedUnref b(gate);

        ComputeWithGate(ctx, gate, done);
    }

  protected:

    virtual void ComputeWithGate(OpKernelContext *ctx, GateType *gate, DoneCallback done) = 0;
  };

    template <typename GateType>
    class SynchronousGateAccessOp : public OpKernel {
    public:
        explicit SynchronousGateAccessOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

        void Compute(OpKernelContext *ctx) final {
            GateType *gate;
            OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "handle", &gate));
            core::ScopedUnref b(gate);

            ComputeWithGate(ctx, gate);
        }

    protected:

        virtual void ComputeWithGate(OpKernelContext *ctx, GateType *gate) = 0;
    };
} // namespace tensorflow {

