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

#include "buffer.h"

namespace tensorflow {
  class BufferPair {
  private:
    Buffer index_, data_;

  public:
    decltype(index_) &index();
    decltype(data_) &data();

    void reset();

    void reserve_size(const size_t total_data_size, const size_t total_num_records);
  };
} // namespace tensorflow {
