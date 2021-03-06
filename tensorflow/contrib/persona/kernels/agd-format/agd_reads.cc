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
#include "agd_reads.h"

#include <utility>

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;

  AGDReadResource::AGDReadResource(size_t num_records, DataContainer *bases, DataContainer *quals, DataContainer *meta) :
    bases_(bases), quals_(quals), meta_(meta), num_records_(num_records)
  {
    // TODO probably chain this with the other constructor
    auto idx_offset = num_records * sizeof(RelativeIndex);
    auto b = bases->get()->data();
    base_idx_ = reinterpret_cast<const RelativeIndex*>(b);
    base_data_ = b + idx_offset;

    auto q = quals->get()->data();
    qual_idx_ = reinterpret_cast<const RelativeIndex*>(q);
    qual_data_ = q + idx_offset;

    auto m = meta->get()->data();
    meta_idx_ = reinterpret_cast<const RelativeIndex*>(m);
    meta_data_ = m + idx_offset;
  }

  AGDReadResource::AGDReadResource(size_t num_records, DataContainer *bases, DataContainer *quals) : bases_(bases), quals_(quals), num_records_(num_records)
  {
    auto idx_offset = num_records * sizeof(RelativeIndex);
    auto b = bases->get()->data();
    base_idx_ = reinterpret_cast<const RelativeIndex*>(b);
    base_data_ = b + idx_offset;

    auto q = quals->get()->data();
    qual_idx_ = reinterpret_cast<const RelativeIndex*>(q);
    qual_data_ = q + idx_offset;
  }

  AGDReadResource&
  AGDReadResource::operator=(AGDReadResource &&other)
  {
    bases_ = other.bases_;
    quals_ = other.quals_;
    meta_ = other.meta_;

    base_idx_ = other.base_idx_;
    qual_idx_ = other.qual_idx_;
    meta_idx_ = other.meta_idx_;

    base_data_ = other.base_data_;
    qual_data_ = other.qual_data_;
    meta_data_ = other.meta_data_;

    num_records_ = other.num_records_;
    record_idx_ = other.record_idx_;

    other.bases_ = nullptr;
    other.quals_ = nullptr;
    other.meta_ = nullptr;

    other.base_data_ = nullptr;
    other.qual_data_ = nullptr;
    other.meta_data_ = nullptr;

    other.base_idx_ = nullptr;
    other.qual_idx_ = nullptr;
    other.meta_idx_ = nullptr;
    return *this;
  }

  bool AGDReadResource::reset_iter()
  {
    record_idx_ = 0;
    auto idx_offset = num_records_ * sizeof(RelativeIndex);
    base_data_ = bases_->get()->data() + idx_offset;
    qual_data_ = quals_->get()->data() + idx_offset;
    if (has_metadata()) {
      meta_data_ = meta_->get()->data() + idx_offset;
    }

    return true;
  }

  bool AGDReadResource::has_metadata()
  {
    return meta_ != nullptr;
  }

  Status AGDReadResource::get_next_record(Read &snap_read)
  {
    if (record_idx_ < num_records_) {
      auto base_len = base_idx_[record_idx_];
      auto *bases = base_data_;
      base_data_  += base_len;

      auto qual_len = qual_idx_[record_idx_++];
      auto *qualities = qual_data_;
      qual_data_ += qual_len;

      snap_read.init(nullptr, 0, bases, qualities, base_len);

      return Status::OK();
    } else {
      return ResourceExhausted("agd record container exhausted");
    }
  }

  Status AGDReadResource::get_next_record(const char** bases, size_t* bases_len,
                         const char** quals, const char** meta,
                         size_t* meta_len) {
    return Internal("unimplemented");
  }

  Status AGDReadResource::get_next_record(const char** bases, size_t* bases_len,
        const char** quals) {
    if (record_idx_ < num_records_) {
      *bases_len = base_idx_[record_idx_];
      *bases = base_data_;
      base_data_  += *bases_len;

      auto qual_len = qual_idx_[record_idx_++];
      *quals = qual_data_;
      qual_data_ += qual_len;

      return Status::OK();
    } else {
      return ResourceExhausted("agd record container exhausted");
    }
  }


  void AGDReadResource::release() {
    if (bases_) {
      bases_->release();
    }
    if (quals_) {
      quals_->release();
    }
    if (meta_) {
      meta_->release();
    }
    sub_resources_.clear();
  }

  Status AGDReadResource::get_next_subchunk(ReadResource **rr, vector<BufferPair*>& b) {
    //size_t idx = sub_resource_index_;
    auto a = sub_resource_index_.fetch_add(1, memory_order_relaxed);
    if (a >= sub_resources_.size()) {
      return ResourceExhausted("No more AGD subchunks");
    }
      //decltype(idx) next;
      //do {
      //  idx = sub_resource_index_;
      //  next = idx+1;
      //  // weak has a few false positives, but is better for loops, according to the spec
      //} while (!sub_resource_index_.compare_exchange_weak(idx, next));
    *rr = &sub_resources_[a];
    b.clear();
    for (auto bl : buffer_lists_)
      b.push_back(&(*bl)[a]);

    return Status::OK();
  }

  Status AGDReadResource::split(size_t chunk, vector<BufferList*>& bl) {
    sub_resources_.clear();

    reset_iter(); // who cares doesn't die for now

    decltype(base_data_) base_start = base_data_, qual_start = qual_data_, meta_start = nullptr;
    decltype(chunk) max_range;
    for (decltype(num_records_) i = 0; i < num_records_; i += chunk) {
      //AGDReadSubResource a(*this, i, CHUNK_SIZE, )
      max_range = i + chunk;
      if (max_range > num_records_) {
        // deals with the tail
        max_range = num_records_;
      }

      sub_resources_.push_back(AGDReadSubResource(*this, i, max_range, base_start, qual_start, meta_start));

      // actually advance the records
      for (decltype(i) j = i; j < max_range; ++j) {
        base_start += base_idx_[j];
        qual_start += qual_idx_[j];
      }
    }
    sub_resource_index_.store(0, memory_order_relaxed);
    buffer_lists_ = bl;
    for (auto b : buffer_lists_) {
      b->resize(sub_resources_.size());
    }
    return Status::OK();
  }

  AGDReadSubResource::AGDReadSubResource(const AGDReadResource &parent_resource,
      size_t index_offset, size_t max_idx,
      const char *base_data_offset, const char *qual_data_offset, const char *meta_data_offset) : start_idx_(index_offset), max_idx_(max_idx), current_idx_(index_offset),
  base_idx_(parent_resource.base_idx_),
  qual_idx_(parent_resource.qual_idx_),
  meta_idx_(parent_resource.meta_idx_),
  base_data_(base_data_offset), base_start_(base_data_offset),
  qual_data_(qual_data_offset), qual_start_(qual_data_offset),
  meta_data_(meta_data_offset), meta_start_(meta_data_offset) {}

  Status AGDReadSubResource::get_next_record(Read &snap_read) {
    if (current_idx_ < max_idx_) {
      auto base_len = base_idx_[current_idx_];
      auto *bases = base_data_;
      base_data_  += base_len;

      auto qual_len = qual_idx_[current_idx_++];
      auto *qualities = qual_data_;
      qual_data_ += qual_len;
      snap_read.init(nullptr, 0, bases, qualities, base_len);

      return Status::OK();
    } else {
      return ResourceExhausted("agd record container exhausted");
    }
  }

  Status AGDReadSubResource::get_next_record(const char** bases, size_t* bases_len,
        const char** quals) {
    if (current_idx_ < max_idx_) {
      *bases_len = base_idx_[current_idx_];
      *bases = base_data_;
      base_data_  += *bases_len;

      auto qual_len = qual_idx_[current_idx_++];
      *quals = qual_data_;
      qual_data_ += qual_len;

      return Status::OK();
    } else {
      return ResourceExhausted("agd record container exhausted");
    }
  }

  Status AGDReadSubResource::get_next_record(const char** bases, size_t* bases_len,
                                          const char** quals, const char** meta,
                                          size_t* meta_len) {
    return Internal("unimplemented");
  }

  bool AGDReadSubResource::reset_iter() {
    base_data_ = base_start_;
    qual_data_ = qual_start_;
    meta_data_ = meta_start_;
    current_idx_ = start_idx_;
    return true;
  }

  size_t AGDReadResource::num_records() {
    return num_records_;
  }

  size_t AGDReadSubResource::num_records() {
    return max_idx_ - start_idx_;
  }

} // namespace tensorflow {
