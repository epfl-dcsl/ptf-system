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
#include "tensorflow/core/lib/core/errors.h"

#include <vector>
#include <zlib.h>

namespace tensorflow {

  Status decompressGZIP(const char* segment,
                        const std::size_t segment_size,
                        std::vector<char> &output);

  Status decompressGZIP(const char* segment,
                        const std::size_t segment_size,
                        Buffer *output);

  Status compressGZIP(const char* segment,
                      const std::size_t segment_size,
                      std::vector<char> &output);

  class AppendingGZIPCompressor
  {
  public:
    AppendingGZIPCompressor(Buffer &output);

    ~AppendingGZIPCompressor();

    // reinitializes the stream
    Status init();

    Status appendGZIP(const char* segment,
                      const std::size_t segment_size);

    // closes the stream
    Status finish(); // somehow flush

  private:
    z_stream stream_ = {0};
    bool done_ = false;
    Buffer &output_;

    void ensure_extend_capacity(std::size_t capacity);
  };

} // namespace tensorflow
