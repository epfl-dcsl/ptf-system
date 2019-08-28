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
#include "buffer_pair.h"
#include "format.h"

namespace tensorflow {

  Buffer& BufferPair::index() {
    return index_;
  }

  Buffer& BufferPair::data() {
    return data_;
  }

  void BufferPair::reset() {
    index_.reset();
    data_.reset();
  }

  void BufferPair::reserve_size(const size_t total_data_size, const size_t total_num_records) {
    // This goofy printout is because otherwise reserve will be optimized away
    // and won't actually touch any pages
    auto a = index_.reserve(total_num_records * sizeof(format::RelativeIndex), true);
    a += data_.reserve(total_data_size, true);
    if (a == 2) {
      LOG(INFO) << "unlikely";
    }
  }

} // namespace tensorflow {
