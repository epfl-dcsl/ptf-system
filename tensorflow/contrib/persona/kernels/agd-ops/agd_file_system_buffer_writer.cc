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
#include "agd_file_system_writer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

namespace tensorflow {
  using namespace std;

  class AGDFileSystemBufferWriter : public AGDFileSystemWriterBase {
  public:
    AGDFileSystemBufferWriter(OpKernelConstruction *ctx) : AGDFileSystemWriterBase(ctx) {
      bool compress;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("compressed", &compress));
      header_.compression_type = compress ? format::CompressionType::GZIP : format::CompressionType::UNCOMPRESSED;
    }

  protected:

    Status WriteResource(OpKernelContext *ctx, FILE *f, const std::string &container, const std::string &name) override {
      ResourceContainer<Data> *column;
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(container, name, &column));

      core::ScopedUnref column_releaser(column);
      {
        ResourceReleaser<Data> pool_releaser(*column);
          auto *b = column->get();
          {
              DataReleaser dr(*b);
              TF_RETURN_IF_ERROR(WriteData(f, b->data(), b->size()));
          }
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("AGDFileSystemBufferWriter").Device(DEVICE_CPU), AGDFileSystemBufferWriter);
} // namespace tensorflow {
