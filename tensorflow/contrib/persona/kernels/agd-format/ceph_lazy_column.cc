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
#include "ceph_lazy_column.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"

namespace tensorflow {

    using namespace std;
    using namespace errors;

    CephLazyColumn::CephLazyColumn(librados::IoCtx &&ctx, const size_t records_per_segment, const size_t num_segments) :
        buffers_(num_segments, vector<char>()), num_segments_(num_segments),
        records_per_segment_(records_per_segment), io_ctx_(ctx) { }

    void CephLazyColumn::Release() {
        next_buffer_ = 0;
        segments_.clear();
        index_ = nullptr;
        header_ = nullptr;
        num_records_ = 0;
        if (delete_after_read_) {
            auto ret = io_ctx_.remove(key_);
            if (ret != 0) {
                throw Internal("UNable to remove key on CephLazyColumn Release");
            }
        }
        key_.clear();
    }

    size_t CephLazyColumn::NumRecords() const {
        return num_records_;
    }

    bool CephLazyColumn::GetNextRecord(const char **const data, size_t *const size) {
        while (not segments_.empty()) {
            auto &segment = segments_.front();
            if (segment.waiting()) {
                auto s = segment.WaitForReady();
                if (not s.ok()) {
                    throw runtime_error(s.error_message());
                }
            }
            auto valid = segment.GetNextRecord(data, size);
            if (valid) {
                return true;
            } else {
                segments_.pop_front();
                auto status = CreateSegment();
                if (not status.ok() and not IsResourceExhausted(status)) {
                    throw runtime_error(status.error_message());
                }
            }
        }
        DCHECK_EQ(next_index_to_read_, num_records_);
        return false;
    }

    bool CephLazyColumn::PeekNextRecord(const char **const data, size_t *const size) {
        while (not segments_.empty()) {
            auto &segment = segments_.front();
            if (segment.waiting()) {
                auto s = segment.WaitForReady();
                if (not s.ok()) {
                    throw runtime_error(s.error_message());
                }
            }
            auto valid = segment.PeekNextRecord(data, size);
            if (valid) {
                return true;
            } else {
                segments_.pop_front();
                auto status = CreateSegment();
                // Need to die here with an exception
                if (not status.ok() and not IsResourceExhausted(status)) {
                    throw runtime_error(status.error_message());
                }
            }
        }
        DCHECK_EQ(next_index_to_read_, num_records_);
        return false;
    }

    Status
    CephLazyColumn::GetRecordAt(size_t index, const char **const data, size_t *const size) {
        return Unimplemented("Indexed access is not yet available for Lazy Ceph reading");
    }

    Status CephLazyColumn::Initialize(const std::string &key, const std::string &name_space, bool delete_after_read) {
        DCHECK(segments_.empty());
        key_ = key;
        next_index_to_read_ = 0;
        io_ctx_.set_namespace(name_space);

        TF_RETURN_IF_ERROR(InitializeMetadata());

        for (remove_const<decltype(num_segments_)>::type i = 0; i < num_segments_; ++i) {
            auto status = CreateSegment();
            if (not status.ok()) {
                if (IsResourceExhausted(status)) {
                    break;
                } else {
                    return status;
                }
            }
        }

        delete_after_read_ = delete_after_read;
        return Status::OK();
    }

    Status CephLazyColumn::InitializeMetadata() {
        static const auto file_header_size = sizeof(format::FileHeader);
        if (header_and_index_.size() < file_header_size) {
            header_and_index_.resize(file_header_size);
        }

        size_t file_size;
        time_t pmtime;
        auto ret = io_ctx_.stat(key_, &file_size, &pmtime);
        if (ret != 0) {
            return Internal("CephLazyColumn: io_ctx_.stat('", key_, "') returned non-0 code ", ret);
        }

        if (file_size < file_header_size) {
            return Internal("CephLazyColumn: Need at least ", file_header_size, " bytes to read the header of '", key_, "', but only have ", file_size);
        }

        librados::bufferlist read_buf;
        read_buf.push_back(ceph::buffer::create_static(file_header_size, &header_and_index_[0]));
        ret = io_ctx_.read(key_, read_buf, file_header_size, 0);
        if (ret != file_header_size) {
            return Internal("CephLazyColumn: Only read ", ret, " bytes for the header of '", key_, "' instead of requested ", file_header_size);
        }

        header_ = reinterpret_cast<const format::FileHeader*>(header_and_index_.data());
        num_records_ = header_->last_ordinal - header_->first_ordinal;
        if (num_records_ < 1) {
            return Internal("Got non-positive number of records in a chunk: ", num_records_);
        }
        size_t index_size_in_bytes = num_records_ * sizeof(format::RelativeIndex);
        size_t total_metadata_size = index_size_in_bytes + file_header_size;

        if (file_size < total_metadata_size) {
            return Internal("CephLazyColumn: Total metadata size for '", key_, "' must be ", total_metadata_size, ", but total file size is ", file_size);
        }

        if (header_and_index_.size() < total_metadata_size) {
            header_and_index_.resize(total_metadata_size);
        }

        read_buf.clear();
        read_buf.push_back(ceph::buffer::create_static(index_size_in_bytes, &header_and_index_.at(file_header_size)));
        ret = io_ctx_.read(key_, read_buf, index_size_in_bytes, file_header_size);
        if (ret != index_size_in_bytes) {
            return Internal("CephLazyColumn: Expected to read ", index_size_in_bytes, " for index of '", key_, "', but only read ", ret);
        }

        // Need to reassign here in case the resize call above changed the pointer to data
        header_ = reinterpret_cast<const format::FileHeader*>(header_and_index_.data());
        index_ = reinterpret_cast<const format::RelativeIndex *>(&header_and_index_[sizeof(format::FileHeader)]);
        absolute_index_.clear();
        size_t offset = total_metadata_size;
        for (size_t i = 0; i < num_records_; ++i) {
            absolute_index_.emplace_back(offset);
            offset += index_[i];
        }
        DCHECK_EQ(absolute_index_.size(), num_records_);

        return Status::OK();
    }

    CephLazyColumn::~CephLazyColumn() {
        io_ctx_.close();
    }

    Status CephLazyColumn::CreateSegment() {
        if (next_index_to_read_ >= num_records_) {
            return ResourceExhausted("Next idx: ", next_index_to_read_, ", num_records: ", num_records_);
        }
        if (segments_.size() == num_segments_) {
            return Internal("Caller needs to check before overrunning segment max (", num_segments_, ")");
        }

        auto remaining_records = min(records_per_segment_, num_records_ - next_index_to_read_);

        auto &buffer = buffers_.at(next_buffer_);
        auto byte_start = absolute_index_.at(next_index_to_read_);

        // need to do it weird like this to handle last index case
        auto end_index = next_index_to_read_+remaining_records-1;
        auto byte_end = absolute_index_.at(end_index) + index_[end_index];

        auto num_bytes = byte_end - byte_start;
        if (buffer.size() < num_bytes) {
            buffer.resize(num_bytes);
        }
        DCHECK_GT(num_bytes, 0);

        try {
            segments_.emplace_back(
                next_index_to_read_, next_index_to_read_+remaining_records,
                index_, buffer,
                byte_start, num_bytes, io_ctx_, key_
            );
        } catch (const std::runtime_error &e) {
            return Internal("Unable to create CephSegment. Got message: ", e.what());
        }
        next_buffer_ = (next_buffer_+1) % buffers_.size();

        next_index_to_read_ += remaining_records;

        return Status::OK();
    }

    void CephLazyColumn::Reset() {
        throw runtime_error("CephLazyColumn doesn't support resetting.");
    }

    Status CephLazyColumn::GetRecordId(std::string &dest) const {
        if (key_.empty()) {
            return Internal("Attempting to get record ID from uninitialized CephLazyColumn");
        }
        DCHECK_NE(header_, nullptr);
        DCHECK_GT(header_and_index_.size(), sizeof(remove_pointer<decltype(header_)>::type));
        dest.assign(&header_->string_id[0], strnlen(&header_->string_id[0], sizeof(header_->string_id)));

        return Status::OK();
    }

    Status CephLazyColumn::CephSegment::WaitForReady() {
        if (read_completion_ != nullptr) {
            read_completion_->wait_for_complete();
            auto ret = read_completion_->get_return_value();
            if (ret != num_bytes_) {
                return Internal("CephSegment: aio_read completion returned ", ret, " when ", num_bytes_, " was expected");
            }
            read_completion_->release();
            read_completion_ = nullptr;
        } else {
            LOG(WARNING) << "CephLazyColumn::Segment waiting more than once for read to complete. Check your code paths!";
        }
        return Status::OK();
    }

    CephLazyColumn::CephSegment::CephSegment(std::size_t start_index, std::size_t end_index,
                                                 const format::RelativeIndex *index, std::vector<char> &buffer,
                                                 const std::size_t byte_offset, const std::size_t num_bytes,
                                                 librados::IoCtx &io_ctx, const std::string &key)
            :
    buffer_(buffer), current_index_(start_index), end_index_(end_index),
    index_(index), num_bytes_(num_bytes) {
        if (buffer_.size() < num_bytes) {
            buffer_.resize(num_bytes);
        }
        buffer_list_.push_back(ceph::buffer::create_static(num_bytes, buffer_.data()));
        read_completion_ = librados::Rados::aio_create_completion();
        auto ret = io_ctx.aio_read(key, read_completion_, &buffer_list_, num_bytes, byte_offset);
        if (ret != 0) {
            throw runtime_error("Got non-0 return when creating aio_read");
        }
    }

    bool CephLazyColumn::CephSegment::GetNextRecord(const char **const data, size_t *const size) {
        if (PeekNextRecord(data, size)) {
            segment_index_ += index_[current_index_++];
            return true;
        }
        return false;
    }

    bool CephLazyColumn::CephSegment::PeekNextRecord(const char **const data, size_t *const size) {
        DCHECK_EQ(read_completion_, nullptr);
        if (current_index_ < end_index_) {
            *size = index_[current_index_];
            *data = &buffer_[segment_index_];
            return true;
        }
        return false;
    }

    bool CephLazyColumn::CephSegment::waiting() const noexcept {
        return read_completion_ != nullptr;
    }

    class CephLazyColumnPoolOp : public ReferencePoolOp<CephLazyColumn, Column> {
    public:
    TF_DISALLOW_COPY_AND_ASSIGN(CephLazyColumnPoolOp);

        explicit CephLazyColumnPoolOp(OpKernelConstruction* ctx) : ReferencePoolOp<CephLazyColumn, Column>(ctx) {
            string user_name, cluster_name, ceph_conf;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_name", &pool_name_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("ceph_conf_path", &ceph_conf));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("records_per_segment", &records_per_segment_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("num_segments", &num_segments_));
            DCHECK_GT(records_per_segment_, 0);
            DCHECK_GT(num_segments_, 0);
            auto ret = cluster_.init2(user_name.c_str(), cluster_name.c_str(), 0);
            OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster init2\nUsername: ", user_name, "\nCluster Name: ", cluster_name, "\nReturn code: ", ret));

            ret = cluster_.conf_read_file(ceph_conf.c_str());

            OP_REQUIRES(ctx, ret == 0, Internal("Ceph conf file at '", ceph_conf, "' returned ", ret, " when attempting to open"));
            ret = cluster_.connect();
            OP_REQUIRES(ctx, ret == 0, Internal("Cluster connect returned: ", ret));
        }

        ~CephLazyColumnPoolOp() override {
            cluster_.shutdown();
        }

    protected:
        unique_ptr<CephLazyColumn> CreateObject() override {

            librados::IoCtx io_ctx;
            auto ret = cluster_.ioctx_create(pool_name_.c_str(), io_ctx);
            if (ret != 0) {
                LOG(ERROR) << name() << ": io_ctx_create(" << pool_name_ << ") returned " << ret;
                return nullptr;
            }
            return unique_ptr<CephLazyColumn>(new CephLazyColumn(
                    move(io_ctx), records_per_segment_, num_segments_
            ));
        }
    private:
        librados::Rados cluster_;
        string pool_name_;
        int records_per_segment_, num_segments_;
    };

    REGISTER_KERNEL_BUILDER(Name("CephLazyColumnPool").Device(DEVICE_CPU), CephLazyColumnPoolOp);
}
