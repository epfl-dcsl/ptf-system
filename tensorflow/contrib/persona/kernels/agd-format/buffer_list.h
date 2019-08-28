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

#include "buffer_pair.h"

namespace tensorflow {
    class BufferList {
    private:
      std::vector<BufferPair> buf_list_;

      void reset_all();

      std::size_t size_ = 0;

    public:
      BufferPair& operator[](std::size_t index);
      std::size_t size() const;
      void resize(std::size_t size);
      void reset();
    };
} // namespace tensorflow {
