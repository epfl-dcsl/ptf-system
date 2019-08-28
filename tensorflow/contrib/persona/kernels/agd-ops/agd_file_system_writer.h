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
namespace tensorflow {
  class AGDFileSystemWriterBase : public AGDWriterBase {
  public:
    AGDFileSystemWriterBase(OpKernelConstruction *ctx);
    void Compute(OpKernelContext* ctx) override final;
  protected:

    virtual Status WriteResource(OpKernelContext *ctx, FILE *f, const std::string &container,
                                 const std::string &name) = 0;

    Status WriteData(FILE *f, const char* data, size_t size);
  };
} // namespace tensorflow {
