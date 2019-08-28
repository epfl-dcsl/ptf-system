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

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/Read.h"
#include "buffer_list.h"
#include <vector>
#include <memory>
#include <atomic>

namespace tensorflow {

  class ReadResource {
  public:
    // TODO how to declare properly?
    virtual ~ReadResource();
    /*
      Iterators over the possible input read data in this resource.

      It is the subclassing class's responsibility to set unavailable fields to 0 / null respectively
     */

    virtual Status get_next_record(Read &snap_read) = 0;
    
    virtual Status get_next_record(const char** bases, size_t* bases_len,
        const char** quals) = 0;

    virtual Status get_next_record(const char** bases, size_t* bases_len,
                                   const char** quals, const char** meta,
                                   size_t* meta_len) = 0;

    virtual std::size_t num_records();

    // Resets the iterator, and returns `true` only if the iterator was successfully reset
    // Non-reset supporting iterators may return false
    virtual bool reset_iter();

    virtual void release();

    // Only valid if the subclass implements subchunks
    virtual Status split(std::size_t chunk, std::vector<BufferList*>& bl);

    virtual Status get_next_subchunk(ReadResource **rr, std::vector<BufferPair*>& b);
   
    // sometimes we don't need a buffer
    //virtual Status get_next_subchunk(ReadResource **rr);
  };

  class ReadResourceReleaser
  {
  public:
    ReadResourceReleaser(ReadResource &r);
    ~ReadResourceReleaser();

  private:
    ReadResource &rr_;
  };

} // namespace tensorflow {
