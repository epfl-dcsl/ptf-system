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

#include <memory>
#include <string>

#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
  class PosixMappedRegion : public ReadOnlyMemoryRegion {
  private:
    PosixMappedRegion(const void* data, uint64 size);
  PosixMappedRegion(const void* data, uint64 size, const std::string &&file_path);

  public:
    // only move semantics
    PosixMappedRegion(PosixMappedRegion &&rhs) noexcept = delete;
    PosixMappedRegion& operator=(PosixMappedRegion &&rhs) noexcept = delete;
    TF_DISALLOW_COPY_AND_ASSIGN(PosixMappedRegion);

    ~PosixMappedRegion() override;
    virtual const void* data() override;
    virtual uint64 length() override;

    static
    Status fromFile(const std::string &filepath, const FileSystem &fs, std::unique_ptr<ReadOnlyMemoryRegion> &result,
                    bool synchronous = false, bool delete_on_free = false);

  private:
    const void* data_ = nullptr;
      std::string file_path_;
    uint64 size_ = 0;
  };


  PosixMappedRegion fromFile(const std::string &filepath);
} // namespace tensorflow {
