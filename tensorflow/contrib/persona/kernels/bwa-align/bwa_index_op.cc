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

// Stuart Byma
// Op providing SNAP genome index and genome

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <memory>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "bwa/bwa.h"
#include "bwa/bwamem.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

    class BWAIndexOp : public OpKernel {
    public:
      typedef BasicContainer<bwaidx_t> BWAIndexContainer;

        BWAIndexOp(OpKernelConstruction* context)
            : OpKernel(context), index_handle_set_(false) {
          OP_REQUIRES_OK(context, context->GetAttr("index_location", &index_location_));
          OP_REQUIRES_OK(context, context->GetAttr("ignore_alt", &ignore_alt_));
          struct stat buf;
          auto ret = stat(index_location_.c_str(), &buf);
          LOG(INFO) << "stat returned: " << ret;
          OP_REQUIRES(context, (ret >=  0),
                      Internal("Index location '", index_location_, "' does not appear to exist"));
          OP_REQUIRES_OK(context,
                         context->allocate_persistent(DT_STRING, TensorShape({ 2 }),
                                                      &index_handle_, nullptr));
        }

        void Compute(OpKernelContext* ctx) override {
            mutex_lock l(mu_);
            if (!index_handle_set_) {
                OP_REQUIRES_OK(ctx, SetIndexHandle(ctx, index_location_));
            }
            ctx->set_output_ref(0, &mu_, index_handle_.AccessTensor(ctx));
        }

    protected:
        ~BWAIndexOp() override {
            // If the genome object was not shared, delete it.
            if (index_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
                TF_CHECK_OK(cinfo_.resource_manager()->Delete<BWAIndexContainer>(
                    cinfo_.container(), cinfo_.name()));
            }
        }

    protected:

        ContainerInfo cinfo_;

    private:
        Status SetIndexHandle(OpKernelContext* ctx, string index_location) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
            BWAIndexContainer* bwa_index;

            auto creator = [this, index_location](BWAIndexContainer** index) {
                LOG(INFO) << "loading bwa index at path: " << index_location;
                auto begin = std::chrono::high_resolution_clock::now();

                bwaidx_t* idx = bwa_idx_load_from_shm(index_location_.c_str());
                if (idx == 0) {
                  LOG(INFO) << "doing regular index load!";
                  if ((idx = bwa_idx_load(index_location_.c_str(), BWA_IDX_ALL)) == 0) 
                    return Internal("Failed to load BWA Index"); 
                }                 
                if (ignore_alt_)
                  for (int i = 0; i < idx->bns->n_seqs; ++i)
                    idx->bns->anns[i].is_alt = 0;
               
                // basic container does not pass the deleter yet ...
                /*auto deleter=[&](bwaidx_t* idx){
	                bwa_idx_destroy(idx);
                };*/
                unique_ptr<bwaidx_t> value(idx);
                auto end = std::chrono::high_resolution_clock::now();
                LOG(INFO) << "index load time is: " << ((float)std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count())/1000000000.0f;
                *index = new BWAIndexContainer(move(value));
                return Status::OK();
            };

            TF_RETURN_IF_ERROR(
                cinfo_.resource_manager()->LookupOrCreate<BWAIndexContainer>(
                    cinfo_.container(), cinfo_.name(), &bwa_index, creator));

            auto h = index_handle_.AccessTensor(ctx)->flat<string>();
            h(0) = cinfo_.container();
            h(1) = cinfo_.name();
            index_handle_set_ = true;
            return Status::OK();
        }

        mutex mu_;
        string index_location_;
        bool ignore_alt_ = false;
        PersistentTensor index_handle_ GUARDED_BY(mu_);
        bool index_handle_set_ GUARDED_BY(mu_);
    };

    REGISTER_KERNEL_BUILDER(Name("BWAIndex").Device(DEVICE_CPU), BWAIndexOp);
}  // namespace tensorflow
