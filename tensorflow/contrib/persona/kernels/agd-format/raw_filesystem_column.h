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

#include <memory>

#include "column.h"
#include "format.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"
#include "data.h"

namespace tensorflow {

class RawFileSystemColumn : public Column {
public:
    using DataContainer = ResourceContainer<Data>;

RawFileSystemColumn() = default;
    static Status AssignFromRawFile(ResourceContainer <Data> *raw_file, RawFileSystemColumn &destination,
                                        std::string &record_id);
private:
    RawFileSystemColumn(ResourceContainer <Data> *raw_file, const size_t num_records,
                    const format::RelativeIndex *const index, const char *data);
public:
RawFileSystemColumn& operator=(RawFileSystemColumn &&other) noexcept;
RawFileSystemColumn(RawFileSystemColumn &&other) = delete;

TF_DISALLOW_COPY_AND_ASSIGN(RawFileSystemColumn);

void Reset() override;
size_t NumRecords() const override;

    bool GetNextRecord(const char **const data, size_t *const size) override;

    bool PeekNextRecord(const char **const data, size_t *const size) override;
Status GetRecordAt(size_t index, const char** const data, size_t* const size) override;

    Status TouchAllData() override;

// Release any underlying resources, if they exist
void Release() override;

private:
    bool PeekCurrent(const char **const data, size_t *const size);

std::unique_ptr<DataContainer, std::function<void(DataContainer*)>> raw_file_;
size_t current_record_index_ = 0, num_records_ = 0;
const char* current_record_data_ = nullptr, *data_start_ = nullptr;
const format::RelativeIndex *index_ = nullptr;
};


} // namespace tensorflow {
