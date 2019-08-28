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

#include "tensorflow/core/framework/resource_op_kernel.h"
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
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/GenomeIndex.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

    using GenomeContainer = BasicContainer<GenomeIndex>;
    class GenomeIndexResourceOp : public ResourceOpKernel<GenomeContainer>
    {
    public:
        explicit GenomeIndexResourceOp(OpKernelConstruction *ctx) : ResourceOpKernel<GenomeContainer>(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("genome_location", &genome_location_));
            FileSystem *fs;

            OP_REQUIRES_OK(ctx, ctx->env()->GetFileSystemForFile(genome_location_, &fs));
            OP_REQUIRES_OK(ctx, fs->FileExists(genome_location_));
        }

    private:
        Status CreateResource(GenomeContainer **resource) override {
            LOG(INFO) << "loading genome index";
            auto begin = std::chrono::high_resolution_clock::now();
            unique_ptr<GenomeIndex> value(GenomeIndex::loadFromDirectory(const_cast<char*>(genome_location_.c_str()), true, true));
            auto end = std::chrono::high_resolution_clock::now();
            LOG(INFO) << "genome load time is: " << ((float)std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count())/1000000000.0f;
            *resource = new GenomeContainer(move(value));
            return Status::OK();
        }


        string genome_location_;
    };

    REGISTER_KERNEL_BUILDER(Name("GenomeIndex").Device(DEVICE_CPU), GenomeIndexResourceOp);
}  // namespace tensorflow
