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

#include <cstdint>
#include <memory>
#include <atomic>
#include "tensorflow/core/lib/core/status.h"
#include "read_resource.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "data.h"

namespace tensorflow {

  class FastqResource : public ReadResource {
  public:
    typedef ResourceContainer<Data> FileResource;

    explicit FastqResource(std::shared_ptr<FileResource> &fastq_file,
                           std::shared_ptr<std::atomic<unsigned int>> &use_count,
                           std::shared_ptr<volatile bool> &done,
                           const char *start_ptr, const char *end_ptr, const std::size_t max_records);
    FastqResource() = default;
    FastqResource& operator=(FastqResource &&other);

    Status get_next_record(Read &snap_read) override;

    Status get_next_record(const char** bases, size_t* bases_len, const char** quals) override;

    Status get_next_record(const char** bases, size_t* bases_len, const char** quals,
                           const char** meta, size_t* meta_len) override;

    bool reset_iter() override;

    void release() override;

    std::size_t num_records();

  private:

    void read_line(const char **line_start, std::size_t *line_length, std::size_t skip_length = 0);
    void skip_line();

    std::shared_ptr<FileResource> fastq_file_; // default constructor = nullptr

    // We must use a shared separate atomic because we need to release the file when all are done
    // and we can't rely on the custom destructor trick because FileResource is, like this type
    // a shared resouce, and we can't rely on any ordering of resource deletion
    // when the session is destroyed
    std::shared_ptr<std::atomic<unsigned int>> file_use_count_;
    std::shared_ptr<volatile bool> done_;

    const char *start_ptr_ = nullptr, *end_ptr_ = nullptr, *current_record_ = nullptr;
    std::size_t max_records_ = 0, current_record_idx_ = 0;

    // prevent unintended copying and assignment
    FastqResource(const FastqResource &other) = delete;
    FastqResource& operator=(const FastqResource &other) = delete;
    FastqResource(FastqResource &&other) = delete;
  };
} // namespace tensorflow {
