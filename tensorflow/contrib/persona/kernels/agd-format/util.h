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

#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "data.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column.h"

namespace tensorflow {

  template <typename T>
    inline void safe_reserve(std::vector<T> &v, const std::size_t ideal_length, const size_t additive_increase_bonus_size = 2 * 1024 * 1024) {
    if (v.capacity() < ideal_length) {
      v.reserve(ideal_length + additive_increase_bonus_size);
    }
  }

  void DataResourceReleaser(ResourceContainer<Data> *data);
    void ColumnResourceReleaser(ResourceContainer<Column> *data);
} // namespace tensorflow {
