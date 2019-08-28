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

namespace tensorflow {
  using namespace std;
  using namespace errors;

  AGDFileSystemWriterBase::AGDFileSystemWriterBase(OpKernelConstruction *ctx) : AGDWriterBase(ctx) {}

  void AGDFileSystemWriterBase::Compute(OpKernelContext *ctx) {
    const Tensor *path_t, *resource_t;
    OP_REQUIRES_OK(ctx, ctx->input("path", &path_t));
    OP_REQUIRES_OK(ctx, ctx->input("resource_handle", &resource_t));
    auto path = path_t->scalar<string>()();
    auto resource_vec = resource_t->vec<string>();

    OP_REQUIRES_OK(ctx, SetHeaderValues(ctx));

    //VLOG(INFO) << "opening file with path: " << path;
    FILE *f = fopen(path.c_str(), "w+");
    OP_REQUIRES(ctx, f != nullptr, Internal("Unable to open file at path ", path));

    OP_REQUIRES_OK(ctx, WriteData(f, reinterpret_cast<const char*>(&header_), sizeof(header_)));
    OP_REQUIRES_OK(ctx, WriteResource(ctx, f, resource_vec(0), resource_vec(1)));
    OP_REQUIRES_OK(ctx, SetOutputKey(ctx, path));

    fclose(f);
  }

  Status AGDFileSystemWriterBase::WriteData(FILE *f, const char *data, size_t size) {
      auto ret = fwrite(data, size, 1, f);
      if (ret != 1) {
          fclose(f);
          return Internal("Unable to write file, fwrite return value was ", ret, " with errno: ", errno);
      }
      return Status::OK();
  }
} // namespace tensorflow {
