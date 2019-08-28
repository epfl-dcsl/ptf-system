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

#include "shared_mmap_file_resource.h"

namespace tensorflow {
  using namespace std;

  MemoryMappedFile::MemoryMappedFile(ResourceHandle &&file) : file_(move(file)) {}

  const char* MemoryMappedFile::data() const {
    return reinterpret_cast<const char*>(file_->data());
  }

  size_t MemoryMappedFile::size() const {
    return file_->length();
  }

  void MemoryMappedFile::release() {
    file_.reset();
  }
} // namespace tensorflow {
