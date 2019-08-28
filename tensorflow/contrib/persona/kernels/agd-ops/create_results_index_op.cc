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
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/agd-format/results_index.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column.h"

namespace tensorflow {
    using namespace std;
    using namespace errors;

    namespace {
        const string op_name("ResultsIndexCreator");
    }

    class ResultsIndexCreatorOp : public OpKernel {
    public:
        ResultsIndexCreatorOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
        }

        ~ResultsIndexCreatorOp() {
            core::ScopedUnref a(index_pool_);
        }

        void Compute(OpKernelContext* ctx) override {
            if (index_pool_ == nullptr) {
                OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "index_pool", &index_pool_));
            }
            ResourceContainer<Column> *results_column;
            ResourceContainer<ResultsIndex> *index_column;
            OP_REQUIRES_OK(ctx, GetInput(ctx, &results_column, &index_column));

            auto &column = *results_column->get();
            auto &index = *index_column->get();

            auto num_records = column.NumRecords();
            DCHECK_GT(num_records, 0);

            index.reset();
            index.reserve(num_records);

            const char* data;
            size_t data_sz;

            Alignment current_result;
            while (column.GetNextRecord(&data, &data_sz)) {
                OP_REQUIRES(ctx, current_result.ParseFromArray(data, data_sz),
                            Internal("CreateResultsIndex: Unable to parse record ", data_sz, " from Results column"));
                index.emplace_back(current_result.position());
            }

            column.Reset();
            OP_REQUIRES_OK(ctx, index_column->allocate_output("index", ctx));
            DCHECK_EQ(num_records, index.size());
        }

    private:
        ReferencePool<ResultsIndex> *index_pool_ = nullptr;

        Status GetInput(OpKernelContext *ctx, ResourceContainer<Column> **column_container,
                        ResourceContainer<ResultsIndex> **index_container) const {
            const Tensor *input_data_t;
            TF_RETURN_IF_ERROR(ctx->input("column", &input_data_t));
            auto input_data_v = input_data_t->vec<string>();
            TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(input_data_v(0), input_data_v(1),
                                                               column_container));
            DCHECK_NE(index_pool_, nullptr);
            return index_pool_->GetResource(index_container);
        }
    };

    REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), ResultsIndexCreatorOp);
} // namespace tensorflow {
