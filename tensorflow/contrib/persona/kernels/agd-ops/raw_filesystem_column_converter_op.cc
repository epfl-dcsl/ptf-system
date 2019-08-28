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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/contrib/persona/kernels/agd-format/raw_filesystem_column.h"
#include "tensorflow/contrib/persona/kernels/agd-format/data.h"

namespace tensorflow {

    using namespace std;
    using namespace errors;
    using namespace format;

    namespace {
        const string op_name("RawFileConverter");
    }

    class RawFileSystemConverterOp : public OpKernel {
    public:
        explicit RawFileSystemConverterOp(OpKernelConstruction *context) : OpKernel(context) { }

        ~RawFileSystemConverterOp() {
            core::ScopedUnref unref_listpool(column_pool_);
        }

        void Compute(OpKernelContext *ctx) override {
            if (column_pool_ == nullptr) {
                OP_REQUIRES_OK(ctx, Init(ctx));
            }
            ResourceContainer<Data> *buf;
            const Tensor *input;
            OP_REQUIRES_OK(ctx, ctx->input("data", &input));
            auto data = input->vec<string>();
            OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &buf));
            // Note: the ResourceContainer will unref the data when it releases it, or this kernel will stop everything

            ResourceContainer<RawFileSystemColumn> *container;
            string record_id;
            OP_REQUIRES_OK(ctx, column_pool_->GetResource(&container));
            OP_REQUIRES_OK(ctx, RawFileSystemColumn::AssignFromRawFile(buf, *container->get(), record_id));
            OP_REQUIRES_OK(ctx, container->allocate_output("column", ctx));

            Tensor *record_id_t;
            OP_REQUIRES_OK(ctx, ctx->allocate_output("record_id", {}, &record_id_t));
            record_id_t->scalar<string>()() = record_id;
        }
    private:

        Status Init(OpKernelContext *ctx) {
            return GetResourceFromContext(ctx, "column_pool", &column_pool_);
        }

        ReferencePool<RawFileSystemColumn> *column_pool_ = nullptr;
    };

    REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), RawFileSystemConverterOp);
} // namespace tensorflow {
