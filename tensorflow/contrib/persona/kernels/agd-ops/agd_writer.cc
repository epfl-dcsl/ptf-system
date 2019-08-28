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
#include <cstring>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "agd_writer.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  AGDWriterBase::AGDWriterBase(OpKernelConstruction *ctx) : OpKernel(ctx) {
    using namespace format;
    string record_suffix;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("record_type", &record_suffix));
    RecordType t;
    if (record_suffix.compare("text") == 0) {
      t = RecordType::TEXT;
    } else if (record_suffix.compare("base_compact") == 0) {
      t = RecordType::COMPACTED_BASES;
    } else {
      t = RecordType::STRUCTURED;
    }
    header_.record_type = static_cast<uint8_t>(t);
    // This is just a safe default
    header_.compression_type = format::CompressionType::UNCOMPRESSED;
  }

  Status AGDWriterBase::SetOutputKey(OpKernelContext* ctx, const string &key) {
    Tensor *key_t;
    TF_RETURN_IF_ERROR(ctx->allocate_output("output_path", TensorShape({}), &key_t));
    key_t->scalar<string>()() = key;
    return Status::OK();
  }

  Status AGDWriterBase::SetHeaderValues(OpKernelContext* ctx) {
    const Tensor *record_id_t, *first_ordinal_t, *num_records_t;
    TF_RETURN_IF_ERROR(ctx->input("record_id", &record_id_t));
    TF_RETURN_IF_ERROR(ctx->input("first_ordinal", &first_ordinal_t));
    TF_RETURN_IF_ERROR(ctx->input("num_records", &num_records_t));
    auto &record_id = record_id_t->scalar<string>()();
    uint64_t first_ordinal = first_ordinal_t->scalar<int64>()();
    uint32_t num_records = num_records_t->scalar<int32>()();

    header_.first_ordinal = first_ordinal;
    header_.last_ordinal = first_ordinal + num_records;

    if (record_id != record_id_) {
        auto const max_copy_size = sizeof(format::FileHeader::string_id);
      auto copy_size = min(record_id.size(), max_copy_size);
      strncpy(&header_.string_id[0], record_id.c_str(), copy_size);
        if (copy_size < max_copy_size) {
            header_.string_id[copy_size] = '\0';
        }
        record_id_ = move(record_id);
    }

    return Status::OK();
  }
} // namespace tensorflow {
