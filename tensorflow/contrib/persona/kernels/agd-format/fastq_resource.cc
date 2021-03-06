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
#include <utility>
#include "fastq_resource.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  FastqResource& FastqResource::operator=(FastqResource &&other) {
    fastq_file_ = move(other.fastq_file_);
    file_use_count_ = move(other.file_use_count_);
    done_ = move(other.done_);
    start_ptr_ = other.start_ptr_; other.start_ptr_ = nullptr;
    end_ptr_ = other.end_ptr_; other.end_ptr_ = nullptr;
    current_record_ = other.current_record_; other.current_record_ = nullptr;
    max_records_ = other.max_records_; other.max_records_ = 0;
    current_record_idx_ = other.current_record_idx_; other.current_record_idx_ = 0;
    return *this;
  }

  FastqResource::FastqResource(shared_ptr<FileResource> &fastq_file,
                               shared_ptr<atomic<unsigned int>> &use_count,
                               shared_ptr<volatile bool> &done,
                               const char *start_ptr, const char *end_ptr, const size_t max_records) :
    fastq_file_(fastq_file), file_use_count_(use_count), done_(done),
    start_ptr_(start_ptr), end_ptr_(end_ptr),
    current_record_(start_ptr),
    max_records_(max_records) {
    file_use_count_->fetch_add(1, memory_order::memory_order_release);
  }

  Status FastqResource::get_next_record(const char** bases, size_t* bases_len,
        const char** quals) {
    return Internal("unimplemented");
  }

  Status FastqResource::get_next_record(const char** bases, size_t* bases_len,
                                        const char** quals, const char** meta,
                                        size_t* meta_len) {

    if (!fastq_file_) {
      return Internal("get_next_record called with null data!");
    } else if (current_record_idx_ == max_records_) {
      return ResourceExhausted("no more records in this file");
    }

    read_line(meta, meta_len, 1); // +1 to skip '@'
    read_line(bases, bases_len);
    skip_line();
    read_line(quals, bases_len);
    current_record_idx_++;

    return Status::OK();

  }

  Status FastqResource::get_next_record(Read &snap_read)
  {
    if (!fastq_file_) {
      return Internal("get_next_record called with null data!");
    } else if (current_record_idx_ == max_records_) {
      return ResourceExhausted("no more records in this file");
    }

    const char *meta, *base, *qual;
    size_t meta_len, base_len;

    read_line(&meta, &meta_len, 1); // +1 to skip '@'
    read_line(&base, &base_len);
    skip_line();
    read_line(&qual, &base_len);
    snap_read.init(meta, meta_len, base, qual, base_len);
    current_record_idx_++;

    return Status::OK();
  }

  void FastqResource::read_line(const char **line_start, size_t *line_length, size_t skip_length)
  {
    current_record_ += skip_length;
    *line_start = current_record_;
    size_t record_size = 0;

    for (; *current_record_ != '\n' && current_record_ < end_ptr_;
         record_size++, current_record_++);

    *line_length = record_size;
    current_record_++; // to skip over the '\n'
  }

  void FastqResource::skip_line()
  {
    for (; *current_record_ != '\n' && current_record_ < end_ptr_; current_record_++);
    /* include this check if we want to avoid leaving the pointers in an invalid state
       currently they won't point to anything, so it should be fine
    if (current_record_ < end_ptr_) {
      current_record_++;
    }
    */
    current_record_++; // to skip over the '\n'
  }

  bool FastqResource::reset_iter()
  {
    current_record_ = start_ptr_;
    current_record_idx_ = 0;
    return true;
  }

  void FastqResource::release() {
    auto count = file_use_count_->fetch_sub(1, memory_order::memory_order_acquire);
    if (count == 1 && *done_) {
      fastq_file_->get()->release();
    }
    file_use_count_.reset();
    fastq_file_.reset();
    start_ptr_ = nullptr;
    end_ptr_ = nullptr;
    current_record_ = nullptr;
    max_records_ = 0;
    current_record_idx_ = 0;
  }

  size_t FastqResource::num_records() {
    return max_records_;
  }

} // namespace tensorflow {
