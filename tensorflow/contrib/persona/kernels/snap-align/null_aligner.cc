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
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <vector>
#include <tuple>
#include <thread>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <locale>
#include <pthread.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_list.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/FileFormat.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/concurrent_queue.h"
#include "GenomeIndex.h"
#include "Read.h"
#include "SnapAlignerWrapper.h"
#include "tensorflow/contrib/persona/kernels/agd-format/read_resource.h"

namespace tensorflow {
using namespace std;
using namespace errors;

  namespace {
    void resource_releaser(ResourceContainer<ReadResource> *rr) {
      ResourceReleaser<ReadResource> a(*rr);
      {
        ReadResourceReleaser r(*rr->get());
      }
    }
  }

class NullAlignerOp : public OpKernel {
  public:
    explicit NullAlignerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
      float wt;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("wait_time_secs", &wt));
      wait_time_ = wt * 1000000; // to put into microseconds
    }

    ~NullAlignerOp() override {
      core::ScopedUnref buflist_pool_unref(buflist_pool_);
    }

    Status InitHandles(OpKernelContext* ctx)
    {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));

      /*if (options_->maxSecondaryAlignmentAdditionalEditDistance < 0) {
        num_secondary_alignments_ = 0;
      } else {
        num_secondary_alignments_ = BaseAligner::getMaxSecondaryResults(options_->numSeedsFromCommandLine,
            options_->seedCoverage, MAX_READ_LENGTH, options_->maxHits, index_resource_->get_index()->getSeedLength());
      }*/

      return Status::OK();
    }

    Status GetResultBufferLists(OpKernelContext* ctx)
    {
      ResourceContainer<BufferList> **ctr;
      Tensor* out_t;
      buffer_lists_.clear();
      buffer_lists_.reserve(1);
      TF_RETURN_IF_ERROR(ctx->allocate_output("result_buf_handle", TensorShape({1, 2}), &out_t));
      auto out_matrix = out_t->matrix<string>();
      for (int i = 0; i < 1; i++) {
        TF_RETURN_IF_ERROR(buflist_pool_->GetResource(ctr));
        //core::ScopedUnref a(reads_container);
        (*ctr)->get()->reset();
        buffer_lists_.push_back((*ctr)->get());
        out_matrix(i, 0) = (*ctr)->container();
        out_matrix(i, 1) = (*ctr)->name();
      }

      return Status::OK();
    }

  void Compute(OpKernelContext* ctx) override {
    //LOG(INFO) << "starting compute!";
    if (buflist_pool_ == nullptr) {
      OP_REQUIRES_OK(ctx, InitHandles(ctx));
    }

    auto start = chrono::high_resolution_clock::now();
    ResourceContainer<ReadResource> *reads_container;
    const Tensor *read_input;
    OP_REQUIRES_OK(ctx, ctx->input("read", &read_input));
    auto data = read_input->vec<string>(); // data(0) = container, data(1) = name
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &reads_container));
    core::ScopedUnref a(reads_container);
    auto reads = reads_container->get();

    OP_REQUIRES_OK(ctx, GetResultBufferLists(ctx));

    OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, buffer_lists_));
    vector<BufferPair*> result_buf;
    ReadResource* subchunk_resource = nullptr;
    Read snap_read;
    const char *bases, *qualities;
    size_t bases_len, qualities_len;
    Status io_chunk_status, subchunk_status;
    io_chunk_status = Status::OK();
    io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_buf);
    while (io_chunk_status.ok()) {
        for (subchunk_status = subchunk_resource->get_next_record(snap_read); subchunk_status.ok();
              subchunk_status = subchunk_resource->get_next_record(snap_read)) {
          char size = static_cast<char>(bases_len);
          result_buf[0]->index().AppendBuffer(&size, 1);
          result_buf[0]->data().AppendBuffer(bases, 28);
        }

        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_buf);
    }
    resource_releaser(reads_container);
    auto end = chrono::high_resolution_clock::now();

    auto null_time = chrono::duration_cast<chrono::microseconds>(end - start);
    decltype(wait_time_) extra_wait = wait_time_ - null_time.count();
    if (extra_wait > 0) {
      //LOG(INFO) << "sleeping for " << extra_wait << " microseconds";
      usleep(extra_wait);
      //this_thread::sleep_for(chrono::microseconds(extra_wait));
    }
  }

private:


  ReferencePool<BufferList> *buflist_pool_ = nullptr;
  int subchunk_size_;
  int64_t wait_time_;
  vector<BufferList*> buffer_lists_;


  Status compute_status_;
  TF_DISALLOW_COPY_AND_ASSIGN(NullAlignerOp);
};

  REGISTER_KERNEL_BUILDER(Name("NullAligner").Device(DEVICE_CPU), NullAlignerOp);

}  // namespace tensorflow
