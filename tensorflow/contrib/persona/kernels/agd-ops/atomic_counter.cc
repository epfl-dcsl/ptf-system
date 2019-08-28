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

#include <atomic>

#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
namespace tensorflow {

    using namespace std;
    using namespace errors;

    namespace {
        const string base_name("AtomicCounter");
        }

    using AtomicType = atomic_uint_fast64_t;
    using AtomicCounter = BasicContainer<AtomicType>;

    class AtomicResourceOp : public ResourceOpKernel<AtomicCounter> {
    public:
        explicit AtomicResourceOp(OpKernelConstruction *context) : ResourceOpKernel(context) {}

    private:
        Status CreateResource(AtomicCounter **resource) override {
            *resource = new AtomicCounter(unique_ptr<AtomicType>(new AtomicType(0ul)));
            return Status::OK();
        }
    };

    class WorkWithAtomic : public OpKernel {
    public:
        explicit WorkWithAtomic(OpKernelConstruction *context) : OpKernel(context) {}

        void Compute(OpKernelContext *ctx) override {
            if (counter_ == nullptr) {
                OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "counter", &counter_));
            }
            OP_REQUIRES_OK(ctx, OperateWithCounter(ctx, *counter_->get()));
        }

    protected:

        virtual Status OperateWithCounter(OpKernelContext *ctx, AtomicType &counter) = 0;

    private:
        AtomicCounter *counter_ = nullptr;
    };

    class AtomicCounterIncrementer : public WorkWithAtomic {
    public:
        explicit AtomicCounterIncrementer(OpKernelConstruction *context) : WorkWithAtomic(context) {}

    protected:
        Status OperateWithCounter(OpKernelContext *ctx, AtomicType &counter) override {
            const Tensor *delta_t;
            TF_RETURN_IF_ERROR(ctx->input("delta", &delta_t));
            auto &delta = delta_t->scalar<int64>()();
            counter.fetch_add(delta, memory_order_relaxed);
            return Status::OK();
        }
    };

    class AtomicCounterFetchAndSet : public WorkWithAtomic {
    public:
        explicit AtomicCounterFetchAndSet(OpKernelConstruction *context) : WorkWithAtomic(context) {}

    protected:
        Status OperateWithCounter(OpKernelContext *ctx, AtomicType &counter) override {
            const Tensor *new_value_t;
            TF_RETURN_IF_ERROR(ctx->input("new_value", &new_value_t));

            Tensor *output_t;
            TF_RETURN_IF_ERROR(ctx->allocate_output("stored_value", TensorShape({}), &output_t));

            auto &new_value = new_value_t->scalar<int64>()();
            auto old_value = counter.exchange(new_value, memory_order_relaxed);
            output_t->scalar<int64>()() = old_value;

            return Status::OK();
        }
    };


    REGISTER_KERNEL_BUILDER(Name(base_name.c_str()).Device(DEVICE_CPU), AtomicResourceOp);
    REGISTER_KERNEL_BUILDER(Name((base_name+"Incrementer").c_str()).Device(DEVICE_CPU), AtomicCounterIncrementer);
    REGISTER_KERNEL_BUILDER(Name((base_name+"FetchAndSet").c_str()).Device(DEVICE_CPU), AtomicCounterFetchAndSet);
} // namespace tensorflow{
