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
#include <cstddef>
#include <string.h>
#include "buffer_list.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

  using namespace std;

  void BufferList::resize(size_t size) {
    auto old_size = buf_list_.size();
    if (size > old_size) {
      buf_list_.resize(size);
    }
    reset_all();
    size_ = size;
  }

  size_t BufferList::size() const {
    return size_;
  }

  BufferPair& BufferList::operator[](size_t index) {
    if (index >= size_) {
      LOG(ERROR) << "FATAL: get_at requested index " << index << ", with only " << size_ << " elements. Real size: " << buf_list_.size();
    }
    // using at instead of operator[] because it will error here
    return buf_list_.at(index);
  }

  void BufferList::reset_all() {
    for (auto &b : buf_list_) {
      b.reset();
    }
  }

  void BufferList::reset() {
    reset_all();
    size_ = 0;
  }

} // namespace tensorflow {
