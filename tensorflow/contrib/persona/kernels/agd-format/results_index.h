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

#include <utility>
#include <vector>
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"

namespace tensorflow{

    // ref index and position
    using FastResult = std::pair<int32, int64>;

    class ResultsIndex {
    public:
        void reserve(size_t expected);

        void reset();

        size_t size() const noexcept;

        void emplace_back(FastResult &&fr);

        void emplace_back(const Position &p);

        const FastResult & current_position() const;
        const FastResult * unsafe_current_position() const;
        bool advance();

    private:
        std::vector<FastResult> fast_results_;
        size_t current_result_ = 0;
    };

} // namespace tensorflow{
