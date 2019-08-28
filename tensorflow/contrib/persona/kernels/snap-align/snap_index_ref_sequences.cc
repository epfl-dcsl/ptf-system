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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/GenomeIndex.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/Genome.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  class SnapIndexReferenceSequencesOp : public OpKernel {
    public:
      explicit SnapIndexReferenceSequencesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

      ~SnapIndexReferenceSequencesOp() override {}

      void Compute(OpKernelContext* ctx) override {

        if (index_resource_ == nullptr) {
          OP_REQUIRES_OK(ctx, InitHandles(ctx));
        }

        auto* contigs = genome_->getContigs();
        int num_contigs = genome_->getNumContigs();

        Tensor* refs_t = NULL, *sizes_t = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}),
              &refs_t));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}),
              &sizes_t));

        auto& refs = refs_t->scalar<string>()();
        auto& sizes = sizes_t->scalar<string>()();
        refs = string(contigs[0].name, contigs[0].nameLength);
        sizes = to_string(contigs[0].length);

        for (int i = 1; i < num_contigs; i++) {
          refs += "|";
          sizes += "|";
          refs += string(contigs[i].name, contigs[i].nameLength);
          sizes += to_string(contigs[i].length);
        }

      }

    private:

      Status InitHandles(OpKernelContext* ctx)
      {
        TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "genome_handle", &index_resource_));

        genome_ = index_resource_->get()->getGenome();

        return Status::OK();
      }

      BasicContainer<GenomeIndex> *index_resource_ = nullptr;
      const Genome *genome_ = nullptr;

      TF_DISALLOW_COPY_AND_ASSIGN(SnapIndexReferenceSequencesOp);
  };

  REGISTER_KERNEL_BUILDER(Name("SnapIndexReferenceSequences").Device(DEVICE_CPU), SnapIndexReferenceSequencesOp);

}  // namespace tensorflow
