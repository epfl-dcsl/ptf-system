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
#include "fastq_chunker.h"

namespace tensorflow {
  using namespace std;

  // note: copies the shared ptr and any custom deleter (which we'll use)
  FastqChunker::FastqChunker(shared_ptr<FastqResource::FileResource> &data, const size_t chunk_size) :
    data_(data), chunk_size_(chunk_size) {
    auto *file_data = data->get();
    current_ptr_ = file_data->data();
    end_ptr_ = current_ptr_ + file_data->size();
    file_use_count_ = make_shared<atomic<unsigned int>>(0);
    done_flag_ = make_shared<volatile bool>(false);
    *done_flag_ = false;
  }

  bool FastqChunker::next_chunk(FastqResource &resource) {
    const char *record_base = current_ptr_;
    size_t record_count = 0;
    while (record_count < chunk_size_ && advance_record()) {
      record_count++;
    }

    // happens if the underlying pointer arithmetic detectse that this is already exhausted
    if (record_count == 0) {
      *done_flag_ = true;
      return false;
    }

    //create a fastq resource
    resource = FastqResource(data_, file_use_count_, done_flag_, record_base, current_ptr_, record_count);

    return true;
  }

  // just assume the basic 4-line format for now
  bool FastqChunker::advance_record() {
    for (int i = 0; i < 4; ++i) {
      if (!advance_line()) {
        return false;
      }
    }
    return true;
  }

  bool FastqChunker::advance_line() {
    if (current_ptr_ == end_ptr_) {
      return false;
    } else {
      while (current_ptr_ < end_ptr_ && *current_ptr_ != '\n') {
        current_ptr_++; // yes, you can put this in the 2nd clause expression, but that is confusing
      }

      // in this case, we want to advance OVER the '\n', as that is what caused us to exit the loop
      if (current_ptr_ < end_ptr_) {
        current_ptr_++;
      }
      return true;
    }
  }

} // namespace tensorflow {
