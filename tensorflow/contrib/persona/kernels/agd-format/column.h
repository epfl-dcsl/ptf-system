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
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

class Column
{
public:
    virtual ~Column() = default;
virtual void Reset() = 0;
virtual size_t NumRecords() const = 0;

    virtual bool GetNextRecord(const char **const data, size_t *const size) = 0;
virtual bool PeekNextRecord(const char **const data, size_t *const size) = 0;

    virtual Status GetRecordAt(size_t index, const char** const data, size_t* const size) = 0;

    virtual Status TouchAllData();


// Release any underlying resources, if they exist
virtual void Release();

};

} //namespace tensorflow {
