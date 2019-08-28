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
#include <sys/types.h>
#include "tensorflow/contrib/persona/kernels/agd-format/shared_mmap_file_resource.h"
#include "tensorflow/contrib/persona/kernels/agd-format/memory_region.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  class FileMMapOp : public OpKernel {
  public:
    FileMMapOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("synchronous", &synchronous_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("delete_after_use", &delete_after_use_));
    };

    ~FileMMapOp() {
      core::ScopedUnref unref_pool(ref_pool);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!ref_pool) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "pool_handle", &ref_pool));
      }
      const Tensor *filename_input;
      OP_REQUIRES_OK(ctx, ctx->input("filename", &filename_input));

      auto filename = filename_input->scalar<string>()();

      ResourceContainer<MemoryMappedFile> *mmf;
      OP_REQUIRES_OK(ctx, ref_pool->GetResource(&mmf));

      unique_ptr<ReadOnlyMemoryRegion> rmr;
      FileSystem *fs;
      OP_REQUIRES_OK(ctx, ctx->env()->GetFileSystemForFile(filename, &fs));
      OP_REQUIRES_OK(ctx, PosixMappedRegion::fromFile(filename, *fs, rmr, synchronous_, delete_after_use_));
        *mmf->get() = move(rmr);

      OP_REQUIRES_OK(ctx, mmf->allocate_output("file_handle", ctx));
    }
  private:
    ReferencePool<MemoryMappedFile> *ref_pool = nullptr;
    bool synchronous_, delete_after_use_;
  };

  REGISTER_KERNEL_BUILDER(Name("FileMMap").Device(DEVICE_CPU), FileMMapOp);
} // namespace tensorflow {
