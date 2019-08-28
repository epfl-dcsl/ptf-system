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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_pair.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("BaseBufferConverter");
  }

  class BaseBufferConverterOp  : public OpKernel {
  public:
    BaseBufferConverterOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor *num_records_t, *buffer_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto &num_records = num_records_t->scalar<int32>()();

      ResourceContainer<Data> *buffer_resource;
      OP_REQUIRES_OK(ctx, ctx->input("buffer", &buffer_t));
      auto buffer_info = buffer_t->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(buffer_info(0),
                      buffer_info(1), &buffer_resource));

      core::ScopedUnref a(buffer_resource);
      OP_REQUIRES_OK(ctx, CompactBases(*buffer_resource->get(), num_records));

      // TODO this output can just be forwarded, but I don't know how to do that
      OP_REQUIRES_OK(ctx, buffer_resource->allocate_output("output_buffer", ctx));
    }

  private:
      vector<format::BinaryBases> compact_;

      Status CompactBases(Data &buffer, int32 num_records) {
          format::RelativeIndex* index = reinterpret_cast<format::RelativeIndex*>(buffer.mutable_data());
          // This math gets us past the number of records in the index
          const auto index_offset = num_records * sizeof(format::RelativeIndex);
          DCHECK_NE(buffer.mutable_data(), nullptr);
          auto src_data = buffer.data() + index_offset;
          auto dest_data = buffer.mutable_data() + index_offset;

          size_t total_size = index_offset; // to set the total size of the buffer after we're done
          for (decltype(num_records) i = 0; i < num_records; i++) {
              auto original_entry_size = index[i];
              TF_RETURN_IF_ERROR(format::IntoBases(src_data, original_entry_size, compact_));
              src_data += original_entry_size;
              size_t num_bytes = compact_.size() * sizeof(format::BinaryBases);
              // mempcpy returns a pointer to the NEXT byte
              dest_data = reinterpret_cast<decltype(dest_data)>(mempcpy(dest_data, compact_.data(), num_bytes));
              total_size += num_bytes;
              index[i] = static_cast<format::RelativeIndex>(num_bytes);
          }
          DCHECK_LE(total_size, buffer.size());
          DCHECK_EQ(dest_data, buffer.mutable_data() + total_size);
          return buffer.resize(total_size);
      }

  };
  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), BaseBufferConverterOp);
} // namespace tensorflow {
