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
#pragma once
#include "agd_writer.h"
#include <rados/librados.hpp>
#include <rados/buffer.h>

// the ceph reader will also use this
// #define ASYNC_CEPH_OPS

namespace tensorflow {
  class AGDCephWriterBase : public AGDWriterBase {
  public:
    AGDCephWriterBase(OpKernelConstruction *ctx);
    void Compute(OpKernelContext* ctx) override final;

      virtual ~AGDCephWriterBase();

  protected:
    virtual Status WritePayload(OpKernelContext *ctx, const std::string &container, const std::string &name, const std::string &key,
                                    librados::bufferlist &write_buf_list, librados::IoCtx &io_ctx) = 0;
    Status SendWrite(OpKernelContext *ctx, librados::bufferlist &write_buf_list, const string &key,
                         librados::IoCtx &io_ctx);

  private:
    librados::Rados cluster_;
      std::string pool_name_;
    librados::bufferlist write_buf_list_;
      librados::IoCtx io_ctx;
  };
} // namespace tensorflow {
