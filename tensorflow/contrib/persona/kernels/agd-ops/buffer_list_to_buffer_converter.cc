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
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_list.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  class BufferListToBufferConverterOp : public OpKernel {
  public:
    BufferListToBufferConverterOp(OpKernelConstruction* context) : OpKernel(context) {}

      ~BufferListToBufferConverterOp() {
          core::ScopedUnref unref_pool(buffer_pool_);
      }

    void Compute(OpKernelContext* ctx) override {
        if (!buffer_pool_) {
            OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_pool", &buffer_pool_));
        }

      ResourceContainer<BufferList> *buffer_list_rc;
      const Tensor *input;
      OP_REQUIRES_OK(ctx, ctx->input("buffer_list", &input));
      auto buffer_list_handle = input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(buffer_list_handle(0), buffer_list_handle(1), &buffer_list_rc));


        ResourceContainer<Buffer> *output_buffer_rc;
        OP_REQUIRES_OK(ctx, buffer_pool_->GetResource(&output_buffer_rc));

        auto buffer_list = buffer_list_rc->get();
        auto buffer = output_buffer_rc->get();
        DCHECK_EQ(buffer->size(), 0);

      core::ScopedUnref a(buffer_list_rc);
      {
        ResourceReleaser<BufferList> b(*buffer_list_rc); // make sure destructs first
          auto num_buffer_lists = buffer_list->size();
          for (decltype(num_buffer_lists) i = 0; i < num_buffer_lists; ++i) {
              auto &buffer_pair = (*buffer_list)[i];
              auto &index = buffer_pair.index();
              buffer->AppendBuffer(index.data(), index.size());
          }
          for (decltype(num_buffer_lists) i = 0; i < num_buffer_lists; ++i) {
              auto &buffer_pair = (*buffer_list)[i];
              auto &data = buffer_pair.data();
              buffer->AppendBuffer(data.data(), data.size());
          }

          buffer_list->reset();
        OP_REQUIRES_OK(ctx, output_buffer_rc->allocate_output("buffer", ctx));
      }
    }
  private:
      ReferencePool<Buffer> *buffer_pool_ = nullptr;
  };


REGISTER_KERNEL_BUILDER(Name("BufferListToBufferConverter").Device(DEVICE_CPU), BufferListToBufferConverterOp);
} // namespace tensorflow {
