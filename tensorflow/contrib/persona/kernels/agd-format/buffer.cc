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
#include "buffer.h"
#include <cstring>
#include <cstddef>
#include "data.h"
#include "util.h"
#include <chrono>

namespace tensorflow {

  using namespace std;
  using namespace errors;

  Buffer::Buffer(size_t initial_size, size_t extend_extra) : extend_extra_(extend_extra), size_(0), allocation_(initial_size) {
    // FIXME in an ideal world, allocation_ should be checked to be positive
    buf_.reset(new char[allocation_]());
  }

  bool Buffer::WriteBuffer(const char *content, size_t content_size) {
    if (allocation_ < content_size) {
      allocation_ = content_size + extend_extra_;
      buf_.reset(new char[allocation_]()); // reset() -> old buf will be deleted
    }
    //auto t1 = std::chrono::high_resolution_clock::now();
    memcpy(buf_.get(), content, content_size);
    //auto t2 = std::chrono::high_resolution_clock::now();
    //auto writememcpytime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    //LOG(INFO) << "writebuf memcpy time: " << writememcpytime.count();
    size_ = content_size;
    return true;
  }

  bool Buffer::AppendBuffer(const char *content, std::size_t content_size) {
    auto old_size = size_;
    extend_size(content_size);
    memcpy(&buf_.get()[old_size], content, content_size);
    return true;
  }

  const char* Buffer::data() const {
    return buf_.get();
  }

  char* Buffer::mutable_data() {
    return buf_.get();
  }

  size_t Buffer::size() const {
    return size_;
  }

  void Buffer::reset() {
    size_ = 0;
  }

  char& Buffer::operator[](size_t idx) const {
    return buf_[idx];
  }

  size_t Buffer::reserve(size_t capacity, bool walk_allocation) {
    size_t x = 0;
    if (capacity > allocation_) {
      allocation_ = capacity + extend_extra_;
      decltype(buf_) a(new char[allocation_]());
      memcpy(a.get(), buf_.get(), size_);
      buf_.swap(a);
      if (walk_allocation) {
        const size_t incr = 4096;
        auto data_current = mutable_data();
        size_t current_idx = size_;
        while (current_idx < allocation_) {
          x += data_current[current_idx];
          data_current[current_idx] = 0;
          current_idx += incr;
        }
      }
    }
    return x;
  }

  Status Buffer::resize(size_t total_size) {
    resize_internal(total_size);
    return Status::OK();
  }
    void Buffer::resize_internal(size_t total_size) {
      reserve(total_size);
      size_ = total_size;
    }


  void Buffer::extend_allocation(size_t extend_size) {
    reserve(allocation_ + extend_size);
  }

  void Buffer::extend_size(size_t extend_size) {
    DCHECK_GT(extend_size, 0);
    resize_internal(size_ + extend_size);
  }

  size_t Buffer::capacity() const {
    return allocation_;
  }

    void Buffer::release() {
        reset();
    }

} // namespace tensorflow {
