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
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_pair.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

namespace tensorflow {
  using namespace std;

  class AGDFileSystemBufferPairWriter : public AGDFileSystemWriterBase {
  public:
    AGDFileSystemBufferPairWriter(OpKernelConstruction *ctx) : AGDFileSystemWriterBase(ctx) {}

  protected:
    Status WriteResource(OpKernelContext *ctx, FILE *f, const std::string &container, const std::string &resource_name) override {
      ResourceContainer<BufferPair> *column;
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(container, resource_name, &column));

      core::ScopedUnref column_releaser(column);
      {
        ResourceReleaser<BufferPair> pool_releaser(*column);
        auto *bp = column->get();
        auto &buf_pair = *bp;
        auto &index = buf_pair.index();
        auto &data = buf_pair.data();
          if (index.size() > 0) {
              TF_RETURN_IF_ERROR(WriteData(f, &index[0], index.size()));
              if (data.size() != 0) {// an empty column
                  TF_RETURN_IF_ERROR(WriteData(f, &data[0], data.size()));
              }
          } else {
              DCHECK_EQ(data.size(), 0);
              LOG(INFO) << name() << ": buffer pair is completely empty.";
          }
        buf_pair.reset();
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("AGDFileSystemBufferPairWriter").Device(DEVICE_CPU), AGDFileSystemBufferPairWriter);
} // namespace tensorflow {
