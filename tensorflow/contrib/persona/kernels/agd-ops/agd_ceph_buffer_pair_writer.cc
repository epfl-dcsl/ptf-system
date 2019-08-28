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
#include "agd_ceph_writer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_pair.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

namespace tensorflow {
  using namespace std;

  class AGDCephBufferPairWriter : public AGDCephWriterBase {
  public:
    AGDCephBufferPairWriter(OpKernelConstruction *ctx) : AGDCephWriterBase(ctx) {}

  protected:
    Status WritePayload(OpKernelContext *ctx, const std::string &container, const std::string &name, const std::string &key,
                            librados::bufferlist &write_buf_list, librados::IoCtx &io_ctx) override {
      ResourceContainer<BufferPair> *column;
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(container, name, &column));

      core::ScopedUnref column_releaser(column);
      {
        ResourceReleaser<BufferPair> pool_releaser(*column);
        auto *bp = column->get();
        auto &buf_pair = *bp;
        auto &index = buf_pair.index();
        auto &data = buf_pair.data();
        write_buf_list.push_back(ceph::buffer::create_static(index.size(), const_cast<char*>(&index[0])));
        write_buf_list.push_back(ceph::buffer::create_static(data.size(), const_cast<char*>(&data[0])));
        TF_RETURN_IF_ERROR(SendWrite(ctx, write_buf_list, key, io_ctx));
        buf_pair.reset();
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("AGDCephBufferPairWriter").Device(DEVICE_CPU), AGDCephBufferPairWriter);
} // namespace tensorflow {
