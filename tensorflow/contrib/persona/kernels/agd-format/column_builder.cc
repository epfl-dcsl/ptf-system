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
#include "column_builder.h"
#include <chrono>
#include "util.h"

namespace tensorflow {

  using namespace std;
  using namespace format;

  void AlignmentResultBuilder::AppendEmpty() {
    format::RelativeIndex val = 0;
    index_->AppendBuffer(reinterpret_cast<const char*>(&val), sizeof(RelativeIndex));
  }

  void AlignmentResultBuilder::AppendAlignmentResult(const Alignment &result)
  {
    //LOG(INFO) << "appending alignment result location: " << result.location_ << " mapq: " << result.mapq_ << " flags: "
    //<< result.flag_ << " next: " << result.next_location_ << " template_len: " << result.template_length_ << " cigar: " << var_string;
    size_t size = result.ByteSize();
    scratch_.resize(size);
    result.SerializeToArray(&scratch_[0], size);
    ColumnBuilder::AppendRecord(&scratch_[0], size);
  }
  
  void ColumnBuilder::SetBufferPair(BufferPair* data) {
    data->reset();
    data_ = &data->data();
    index_ = &data->index();
  }

  void ColumnBuilder::AppendRecord(const char* data, const size_t size) {
    if (size > format::MAX_INDEX_SIZE)
      LOG(ERROR) << "WARNING: Appending data larger than " << UINT8_MAX << " bytes not supported by AGD.";
    if (size > 0) // could be a zero record
      data_->AppendBuffer(data, size);
    format::RelativeIndex cSize = static_cast<uint16_t>(size);
    index_->AppendBuffer(reinterpret_cast<const char *>(&cSize), sizeof(cSize));
  }

} // namespace tensorflow {
