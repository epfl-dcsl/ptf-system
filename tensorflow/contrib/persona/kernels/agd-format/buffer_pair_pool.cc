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
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_pair.h"

namespace tensorflow {
  using namespace std;

  class BufferPairPoolOp : public ReferencePoolOp<BufferPair, BufferPair> {
  public:
    BufferPairPoolOp(OpKernelConstruction* ctx) : ReferencePoolOp<BufferPair, BufferPair>(ctx) {
    }

  protected:
    unique_ptr<BufferPair> CreateObject() override {
      return unique_ptr<BufferPair>(new BufferPair());
    }

  private:
    TF_DISALLOW_COPY_AND_ASSIGN(BufferPairPoolOp);
  };

  class PrimedBufferPairPoolOp : public BufferPairPoolOp {
  public:
      PrimedBufferPairPoolOp(OpKernelConstruction* ctx) : BufferPairPoolOp(ctx) {
          OP_REQUIRES_OK(ctx, ctx->GetAttr("num_records", &num_records_));
          OP_REQUIRES_OK(ctx, ctx->GetAttr("record_size", &data_size_));
          data_size_ *= num_records_;
      }
  protected:
      unique_ptr<BufferPair> CreateObject() override {
          auto bp = BufferPairPoolOp::CreateObject();
          if (not bp) {
              throw runtime_error("Unable to allocate BufferPairPool");
          }
          bp->reserve_size(data_size_, num_records_);
          return bp;
      }
  private:
      int32 num_records_, data_size_;
  };

  REGISTER_KERNEL_BUILDER(Name("BufferPairPool").Device(DEVICE_CPU), BufferPairPoolOp);
  REGISTER_KERNEL_BUILDER(Name("PrimedBufferPairPool").Device(DEVICE_CPU), PrimedBufferPairPoolOp);
} // namespace tensorflow {
