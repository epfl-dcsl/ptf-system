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
#include "raw_filesystem_column.h"
#include "parser.h"

namespace tensorflow {
    using namespace std;
    using namespace errors;

namespace {
    void release_data(RawFileSystemColumn::DataContainer *raw_file) {
        core::ScopedUnref su(raw_file);
        {
            ResourceReleaser<Data> rr(*raw_file);
            raw_file->get()->release();
        }
    }
}

RawFileSystemColumn::RawFileSystemColumn(ResourceContainer <Data> *raw_file, const size_t num_records,
                                         const format::RelativeIndex *const index, const char *data) :
    raw_file_(raw_file, release_data), num_records_(num_records), index_(index), current_record_data_(data), data_start_(data) {}

void RawFileSystemColumn::Reset() {
    current_record_index_ = 0;
    current_record_data_ = data_start_;
}

size_t RawFileSystemColumn::NumRecords() const {
    return num_records_;
}

bool RawFileSystemColumn::GetNextRecord(const char **const data, size_t *const size) {
    if (PeekCurrent(data, size)) {
        current_record_data_ += *size;
        current_record_index_++;
        return true;
    }
    return false;
}

bool RawFileSystemColumn::PeekNextRecord(const char **const data, size_t *const size) {
    return PeekCurrent(data, size);
}

Status RawFileSystemColumn::GetRecordAt(size_t index, const char** const data, size_t* const size) {
    return Unimplemented("Haven't implemented individual record access yet");
}

// Release any underlying resources, if they exist
void RawFileSystemColumn::Release() {
    raw_file_.reset();
    current_record_data_ = nullptr;
    index_ = nullptr;
}

    RawFileSystemColumn &RawFileSystemColumn::operator=(RawFileSystemColumn &&other) noexcept {
        raw_file_ = move(other.raw_file_);
        current_record_index_ = other.current_record_index_;
        num_records_ = other.num_records_; DCHECK_GT(num_records_, 0);
        index_ = other.index_; other.index_ = nullptr;
        current_record_data_ = other.current_record_data_; other.current_record_data_ = nullptr;
        data_start_ = other.data_start_; other.data_start_ = nullptr;
        return *this;
    }

    Status RawFileSystemColumn::AssignFromRawFile(ResourceContainer <Data> *raw_file, RawFileSystemColumn &destination,
                                                      std::string &record_id) {
        size_t num_records;
        const format::RelativeIndex *relative_index;
        const char* data_start;
        auto data = raw_file->get();
        TF_RETURN_IF_ERROR(ParseValuesFromHeader(data->data(), data->size(), &num_records, &relative_index, &data_start,
                                                 record_id));

        destination = RawFileSystemColumn(raw_file, num_records, relative_index, data_start);
        return Status::OK();
    }

    bool RawFileSystemColumn::PeekCurrent(const char **const data, size_t *const size) {
        if (current_record_index_ == num_records_) {
            return false;
        }
        *data = current_record_data_;
        *size = index_[current_record_index_];
        return true;
    }

    Status RawFileSystemColumn::TouchAllData() {
        const size_t pg_size = 4096;
        auto real_raw_file = raw_file_->get();
        auto max_size = real_raw_file->size();
        auto data_start = real_raw_file->data();

        char a = 0;
        for (size_t idx = 0; idx < max_size; idx += pg_size) {
            a += data_start[idx];
        }
        if (a == 0) {
            LOG(INFO) << "meh";
        } else {
            LOG(INFO) << "ok";
        }
        return Status::OK();
    }

    class RawFileSystemColumnPoolOp : public ReferencePoolOp<RawFileSystemColumn, Column> {
    public:
        RawFileSystemColumnPoolOp(OpKernelConstruction* ctx) : ReferencePoolOp<RawFileSystemColumn, Column>(ctx) {
        }

    protected:
        unique_ptr<RawFileSystemColumn> CreateObject() override {
            return unique_ptr<RawFileSystemColumn>(new RawFileSystemColumn());
        }
    private:
        TF_DISALLOW_COPY_AND_ASSIGN(RawFileSystemColumnPoolOp);
    };

    REGISTER_KERNEL_BUILDER(Name("RawFileSystemColumnPool").Device(DEVICE_CPU), RawFileSystemColumnPoolOp);
} // namespace tensorflow {
