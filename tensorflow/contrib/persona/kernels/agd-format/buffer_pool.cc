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
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"

namespace tensorflow {
  using namespace std;

  class BufferPoolOp : public ReferencePoolOp<Buffer, Data> {
  public:
    BufferPoolOp(OpKernelConstruction* ctx) : ReferencePoolOp<Buffer, Data>(ctx) {
    }

  protected:
    unique_ptr<Buffer> CreateObject() override {
      return unique_ptr<Buffer>(new Buffer());
    }

  private:
    TF_DISALLOW_COPY_AND_ASSIGN(BufferPoolOp);
  };

  REGISTER_KERNEL_BUILDER(Name("BufferPool").Device(DEVICE_CPU), BufferPoolOp);
} // namespace tensorflow {
