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
#include "results_index.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"

namespace tensorflow{
    using namespace std;

    void ResultsIndex::reserve(size_t expected) {
        if (fast_results_.capacity() < expected) {
            fast_results_.reserve(expected);
        }
    }

    void ResultsIndex::reset() {
        fast_results_.clear();
        current_result_ = 0;
    }

    void ResultsIndex::emplace_back(FastResult &&fr) {
        fast_results_.emplace_back(move(fr));
    }

    void ResultsIndex::emplace_back(const Position &p) {
        emplace_back(move(FastResult(p.ref_index(), p.position())));
    }

    size_t ResultsIndex::size() const noexcept {
        return fast_results_.size();
    }

    const FastResult & ResultsIndex::current_position() const {
        DCHECK_LT(current_result_, fast_results_.size());
        return fast_results_[current_result_];
    }

    const FastResult * ResultsIndex::unsafe_current_position() const {
        DCHECK_LT(current_result_, fast_results_.size());
        return &fast_results_[current_result_];
    }

    bool ResultsIndex::advance() {
        if (current_result_ == fast_results_.size() - 1) {
            return false;
        }
        DCHECK_LT(current_result_, fast_results_.size() - 1); // we're using == for speed
        current_result_++;
        return true;
    }

    class ResultsIndexPoolOp : public ReferencePoolOp<ResultsIndex, ResultsIndex> {
    public:
        ResultsIndexPoolOp(OpKernelConstruction* ctx) : ReferencePoolOp<ResultsIndex, ResultsIndex>(ctx) {
        }

    protected:
        unique_ptr<ResultsIndex> CreateObject() override {
            return unique_ptr<ResultsIndex>(new ResultsIndex());
        }

    private:
        TF_DISALLOW_COPY_AND_ASSIGN(ResultsIndexPoolOp);
    };

    REGISTER_KERNEL_BUILDER(Name("ResultsIndexPool").Device(DEVICE_CPU), ResultsIndexPoolOp);
}
