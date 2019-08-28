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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/agd-format/data.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/ceph_lazy_column.h"

#include "agd_ceph_writer.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  class LazyCephReaderOp : public OpKernel {
  public:
    LazyCephReaderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("delete_after_read", &delete_after_read_));
    }

    ~LazyCephReaderOp() {
      core::ScopedUnref unref_pool(column_pool_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (column_pool_ == nullptr) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "column_pool", &column_pool_));
      }
        const Tensor *key_t, *namespace_t;
        OP_REQUIRES_OK(ctx, ctx->input("key", &key_t));
        OP_REQUIRES_OK(ctx, ctx->input("namespace", &namespace_t));
        auto file_key = key_t->scalar<string>()();
        auto name_space = namespace_t->scalar<string>()();

        ResourceContainer<CephLazyColumn> *rc;
        OP_REQUIRES_OK(ctx, column_pool_->GetResource(&rc));
        auto column = rc->get();

        OP_REQUIRES_OK(ctx, column->Initialize(file_key, name_space, delete_after_read_));
        OP_REQUIRES_OK(ctx, column->GetRecordId(record_id_));
        OP_REQUIRES_OK(ctx, rc->allocate_output("column_handle", ctx));

        Tensor *record_id_t;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("record_id", {}, &record_id_t));
        record_id_t->scalar<string>()() = record_id_;
    }

  private:
      bool delete_after_read_;
      string record_id_;
    ReferencePool<CephLazyColumn> *column_pool_ = nullptr;
  };

  REGISTER_KERNEL_BUILDER(Name("LazyCephReader").Device(DEVICE_CPU), LazyCephReaderOp);

} // namespace tensorflow {
