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
/*
  An op to read from a queue of filename strings, and enqueue multiple shared resources corresponding to each file.
 */

#pragma once

#include <memory>
#include <cstdint>
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "data.h"

namespace tensorflow {
  /*
    Just a convenience class to track a read-only memory region throughout the execution context.
   */

  class MemoryMappedFile : public Data {
  public:
    typedef std::unique_ptr<ReadOnlyMemoryRegion> ResourceHandle;

    virtual const char* data() const override;
    virtual std::size_t size() const override;

    // needed for pool creation
    MemoryMappedFile() = default;
    MemoryMappedFile(ResourceHandle &&file);
    MemoryMappedFile& operator=(MemoryMappedFile &&x) = default;
      ~MemoryMappedFile() override = default;

    void release() override;
    TF_DISALLOW_COPY_AND_ASSIGN(MemoryMappedFile);
  private:
    ResourceHandle file_;
  };

} // namespace tensorflow {
