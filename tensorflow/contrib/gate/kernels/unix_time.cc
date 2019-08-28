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
#include <chrono>
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
    using namespace std;
    using namespace errors;

    namespace {
        const string op_name("UnixTimestamp");
    }

    class UnixTimestampOp : public OpKernel {
    public:
        explicit UnixTimestampOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
        }

        void Compute(OpKernelContext *ctx) override {
            Tensor *t;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &t));
            auto t_val = t->scalar<int64>();

            auto timepoint = chrono::system_clock::now();
            auto duration = timepoint.time_since_epoch();
            auto seconds = chrono::duration_cast<chrono::microseconds>(duration).count();
            t_val() = static_cast<int64>(seconds);
        }
    };
    REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), UnixTimestampOp);
}
