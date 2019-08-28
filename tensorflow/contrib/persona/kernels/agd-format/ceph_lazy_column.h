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

#include "column.h"
#include "format.h"
#include <vector>
#include <rados/librados.hpp>
#include <boost/circular_buffer.hpp>

namespace tensorflow {
    // Note that this is not meant to be used in a thread-unsafe manner
    // Meant to primarily be used in a merge setting
    class CephLazyColumn : public Column {
    public:
        CephLazyColumn(librados::IoCtx &&ctx, const size_t records_per_segment, const size_t num_segments);

        Status Initialize(const std::string &key, const std::string &name_space, bool delete_after_read);

        ~CephLazyColumn() override;

        void Reset() override;

        void Release() override;

        size_t NumRecords() const override;

        bool GetNextRecord(const char **const data, size_t *const size) override;

        bool PeekNextRecord(const char **const data, size_t *const size) override;

        Status GetRecordAt(size_t index, const char **const data, size_t *const size) override;

        Status GetRecordId(std::string &dest) const;

    private:
        class CephSegment {
        public:
            // These own the actual underlying data
            CephSegment(std::size_t start_index, std::size_t end_index,
                        const format::RelativeIndex *index, std::vector<char> &buffer,
                        const std::size_t byte_offset, const std::size_t num_bytes,
                        librados::IoCtx &io_ctx, const std::string &key);

            // TODO change these return the index, too
            bool GetNextRecord(const char **const data, size_t *const size);

            bool PeekNextRecord(const char **const data, size_t *const size);

            Status WaitForReady();

            bool waiting() const noexcept;

            // This object must stay intact upon initialization
            // so that the aio read completion buffer_list_ always points to valid memory
            TF_DISALLOW_COPY_AND_ASSIGN(CephSegment);
            CephSegment(CephSegment &&) = delete;
            CephSegment & operator=(CephSegment &&) = delete;
        private:
            std::vector<char>& buffer_;
            const format::RelativeIndex *index_;

            std::size_t current_index_, segment_index_ = 0;
            const std::size_t end_index_, num_bytes_;

            librados::bufferlist buffer_list_;
            librados::AioCompletion *read_completion_ = nullptr;
        };

        Status CreateSegment();

        std::vector<std::vector<char>> buffers_;
        std::size_t next_buffer_ = 0, next_index_to_read_;
        std::deque<CephSegment> segments_;

        std::vector<char> header_and_index_;
        std::vector<std::size_t> absolute_index_; // index is relative to the file, so it includes the offsets from the headers
        const format::RelativeIndex *index_ = nullptr;
        const format::FileHeader *header_ = nullptr;
        std::size_t num_records_;

        Status InitializeMetadata();

        librados::IoCtx io_ctx_;
        std::string key_; // use key.empty() to determine whether this is set. you already have to do it anyway

        const std::size_t records_per_segment_, num_segments_;
        bool delete_after_read_;
    };
} // namespace tensorflow{
