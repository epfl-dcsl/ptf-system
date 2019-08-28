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
#include <cstdint>
#include <string>
#include "format.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/AlignmentResult.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "buffer_list.h"

namespace tensorflow {

class ColumnBuilder {
public:
    virtual ~ColumnBuilder() {};
    virtual void SetBufferPair(BufferPair* data);

    virtual void AppendRecord(const char* data, const std::size_t size);

protected:
    Buffer *data_ = nullptr, *index_ = nullptr;
};

class AlignmentResultBuilder : public ColumnBuilder {
public:
    using ColumnBuilder::AppendRecord;
    using ColumnBuilder::SetBufferPair;
    /*
      Append the current alignment result to the internal result buffer
      result_size is needed to ensure that the result is the correct size

      The AppendAlignmentResult Methods take a character index, which is passed in from outside to
      enable the aligner kernel to pass in a buffer from a pool, to build up a result in it.

      records_ is used to build up the actual reads, which are appended into the chunk.
    */

  //void AppendAlignmentResult(const SingleAlignmentResult &result, const std::string &var_string, const int flag);

  //void AppendAlignmentResult(const SingleAlignmentResult &result);

  // This is the only one we should use now


  void AppendAlignmentResult(const Alignment &result);
  // sometimes we want to append an empty result
  // e.g. not all reads will generate X secondary alignments (so some columns will have gaps)
  void AppendEmpty();

  //void AppendAlignmentResult(const PairedAlignmentResult &result, const std::size_t result_idx);

  private:
    std::vector<char> scratch_;
  };


} // namespace tensorflow
