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
#include "parser.h"
#include <utility>
#include <array>
#include "compression.h"
#include "util.h"
#include "format.h"
#include "tensorflow/core/platform/mutex.h"
#include <cstdint>

namespace tensorflow {

  using namespace std;
  using namespace errors;

  unsigned char nst_nt4_table[256] = {
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5 /*'-'*/, 4, 4,
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
          4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
  };

  namespace {
    volatile bool table_needs_init_ = true;
    const size_t base_width = 3;
    const size_t num_bases = 512; // 2*(enum_bit_width{3} * base_width{3})
    const auto mask = ~(~0 << format::BinaryBases::base_width);
    array<BaseMapping<base_width>, num_bases> base_table_;
    mutex base_table_mu_;

    template <size_t N>
    BaseMapping<N>
    make_result(size_t key) {
      using namespace format;
      array<char, N> ret;
      ret.fill('\0');
      uint64_t masked;
      bool run = true, valid = true;
      char c;
      size_t usable_characters = 0;
      for (size_t i = 0; run && i < ret.size(); ++i) {
        masked = static_cast<BaseAlphabet>(key & mask);
        switch (masked) {
        default:
          // if a bad value is found, we need to set to a default "bad" value, as below
          run = false;
          valid = false;
          break;
        case BaseAlphabet::A:
          c = 'A';
          break;
        case BaseAlphabet::T:
          c = 'T';
          break;
        case BaseAlphabet::C:
          c = 'C';
          break;
        case BaseAlphabet::G:
          c = 'G';
          break;
        case BaseAlphabet::N:
          c = 'N';
          break;
        case BaseAlphabet::END:
          run = false;
          break;
        }
        if (run) {
          ret[i] = c;
          usable_characters++;
          key >>= BinaryBases::base_width;
        }
      }

      if (!valid) {
        ret.fill('\0');
        usable_characters = 0;
      }

      return BaseMapping<N>(ret, usable_characters);
    }

    void init_table_locked() {
      for (size_t i = 0; i < base_table_.size(); ++i) {
        base_table_[i] = make_result<base_width>(i);
      }
    }

    void init_table() {
      if (table_needs_init_) {
        mutex_lock l(base_table_mu_);
        if (table_needs_init_) {
          init_table_locked();
          table_needs_init_ = false;
        }
      }
    }
  }

  const BaseMapping<3>*
  lookup_triple(const size_t bases) {
    auto b = bases & 0x1ff;
    const auto &a = base_table_[b];
    if (a.get()[0] == 'Z') {
      return nullptr;
    }
    return &a;
  }

  Status RecordParser::ParseNew(const char* data, const std::size_t length, const bool verify, Buffer *result_buffer, uint64_t *first_ordinal, uint32_t *num_records, string &record_id, bool unpack, bool repack)
  {
    using namespace errors;
    using namespace format;
    reset();
    if (unpack and repack) {
      return Internal("Bad arguments! Can't unpack and repack");
    }

    if (length < sizeof(FileHeader)) {
      return Internal("AGDReader::FillBuffer: needed min length ", sizeof(FileHeader),
                      ", but only received ", length);
    }
    auto file_header = reinterpret_cast<const FileHeader*>(data);
    auto record_type = static_cast<RecordType>(file_header->record_type);
    switch (record_type) {
    case RecordType::TEXT:
    case RecordType::STRUCTURED:
    case RecordType::COMPACTED_BASES:
      break;
    default:
      return Internal("Invalid record type ", file_header->record_type);
    }

    auto payload_start = data + file_header->segment_start;
    auto payload_size = length - file_header->segment_start;

    Status status;
    auto compression_type = static_cast<CompressionType>(file_header->compression_type);
    switch (compression_type) {
    case CompressionType::GZIP:
      status = decompressGZIP(payload_start, payload_size, result_buffer);
      break;
    case CompressionType::UNCOMPRESSED:
      if (result_buffer->WriteBuffer(payload_start, payload_size)) {
        status = Internal("Unable to write uncompressed buffer of size ", payload_size);
      }
      break;
    default:
      status = errors::InvalidArgument("Compressed type '", file_header->compression_type, "' doesn't match to any valid or supported compression enum type");
      break;
    }
    const size_t index_size = file_header->last_ordinal - file_header->first_ordinal;
    const size_t index_size_bytes = index_size * sizeof(RelativeIndex);
    TF_RETURN_IF_ERROR(status);

    /*if (result_buffer->size() < index_size * 2) {
      return Internal("FillBuffer: expected at least ", index_size*2, " bytes, but only have ", result_buffer->size());
    }*/ // stuart: removing because we can have 0-length records

    records = reinterpret_cast<const RelativeIndex*>(result_buffer->data());

    if (verify) {
      size_t data_size = 0;
      // This iteration is expensive, which is why this is optional. Run with perf to get an idea
      for (uint64_t i = 0; i < index_size; ++i) {
        data_size += records[i];
      }

      const size_t expected_size = result_buffer->size() - index_size_bytes;
      if (data_size != expected_size) {
        if (data_size < expected_size) {
          return OutOfRange("Expected a file size of ", expected_size, " bytes, but only found ",
                            data_size, " bytes");
        } else {
          return OutOfRange("Expected a file size of ", expected_size, " bytes, but only found ",
                            data_size, " bytes");
        }
      }
    }

    if (unpack and record_type == RecordType::COMPACTED_BASES) {
      conversion_scratch_.reset(); index_scratch_.reset();

      RelativeIndex current_record_length;
      const char* start_ptr = &(*result_buffer)[index_size_bytes];
      const BinaryBases *bases;

      for (uint64_t i = 0; i < index_size; ++i) {
        current_record_length = records[i];
        bases = reinterpret_cast<const BinaryBases*>(start_ptr);
        start_ptr += current_record_length;

        TF_RETURN_IF_ERROR(append(bases, static_cast<size_t>(current_record_length), conversion_scratch_, index_scratch_));
      }

      // append everything in converted_records to the index
      result_buffer->reserve(index_scratch_.size() + conversion_scratch_.size());
      if (not (result_buffer->WriteBuffer(&index_scratch_[0], index_scratch_.size()) and result_buffer->AppendBuffer(&conversion_scratch_[0], conversion_scratch_.size()))) {
        return Internal("Unable to write buffer or index");
      }
    } else if (repack and record_type == RecordType::TEXT) {
      TF_RETURN_IF_ERROR(CompactBases(*result_buffer, index_size));
    }

    *first_ordinal = file_header->first_ordinal;
    *num_records = index_size;
    record_id.assign(&file_header->string_id[0], strnlen(&file_header->string_id[0], sizeof(file_header->string_id)));
    return Status::OK();
  }

  void RecordParser::reset() {
    conversion_scratch_.reset();
    index_scratch_.reset();
  }

  RecordParser::RecordParser() {
    init_table();
  }

  Status ParseValuesFromHeader(const char *const data, const size_t length, size_t *const num_records,
                                 const format::RelativeIndex **relative_index, const char **data_start,
                                 std::string &record_id) {
    using namespace format;
      if (length < sizeof(FileHeader)) {
        return Internal("Needed ", sizeof(FileHeader), " bytes in data, but got ", length);
      }

    auto const header = reinterpret_cast<const FileHeader*>(data);
    if (header->compression_type != UNCOMPRESSED) {
      return Internal("Attempting to lay down a raw unstructured file on an compressed type.");
    }
      auto payload_size = length - header->segment_start;
      auto num_recs = header->last_ordinal - header->first_ordinal;
    decltype(payload_size) expected_index_size = num_recs * sizeof(RelativeIndex);
    if (payload_size < expected_index_size) {
      return Internal("Insufficient size for index. Header specified size of ", expected_index_size, ", but only have ", payload_size);
    }
    *num_records = num_recs;
    *relative_index = reinterpret_cast<const RelativeIndex *>(data+header->segment_start);
    *data_start = data + (header->segment_start + expected_index_size);
      record_id.assign(&header->string_id[0], strnlen(&header->string_id[0], sizeof(header->string_id)));
    return Status::OK();
    }

    Status CompactBases(Data &buffer, int32 num_records) {
        vector<format::BinaryBases> compact;
        format::RelativeIndex* index = reinterpret_cast<format::RelativeIndex*>(buffer.mutable_data());
        // This math gets us past the number of records in the index
        const auto index_offset = num_records * sizeof(format::RelativeIndex);
        DCHECK_NE(buffer.mutable_data(), nullptr);
        auto src_data = buffer.data() + index_offset;
        auto dest_data = buffer.mutable_data() + index_offset;

        size_t total_size = index_offset; // to set the total size of the buffer after we're done
        for (decltype(num_records) i = 0; i < num_records; i++) {
            auto original_entry_size = index[i];
            TF_RETURN_IF_ERROR(format::IntoBases(src_data, original_entry_size, compact));
            src_data += original_entry_size;
            size_t num_bytes = compact.size() * sizeof(format::BinaryBases);
            // mempcpy returns a pointer to the NEXT byte
            dest_data = reinterpret_cast<decltype(dest_data)>(mempcpy(dest_data, compact.data(), num_bytes));
            total_size += num_bytes;
            index[i] = static_cast<format::RelativeIndex>(num_bytes);
        }
        DCHECK_LE(total_size, buffer.size());
        DCHECK_EQ(dest_data, buffer.mutable_data() + total_size);
        return buffer.resize(total_size);
    }
} // namespace tensorflow {
