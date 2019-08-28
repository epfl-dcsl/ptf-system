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

#include <deque>
#include <vector>
#include <functional>
#include <unordered_set>

#include "streaming_gate.h"
#include "tensor_slicing.h"

namespace tensorflow {

// Functionality common to asynchronous BarrierInterface implementations.
    class EgressGate : public StreamingGate {
    public:
        // Args:
        //   component_dtypes: The types of each component in a queue-element tuple.
        //   component_shapes: The shapes of each component in a queue-element tuple,
        //     which must either be empty (if the shapes are not specified) or
        //     or have the same size as component_dtypes.
        //   name: A name to use for the queue.
        EgressGate(const DataTypeVector &component_dtypes, const std::vector<TensorShape> &component_shapes,
                   const string &name, const int32 capacity, const bool limit_upstream);

        using CallbackWithPlainResult = std::function<void(const Tuple&)>;

        Status SupplyDownstreamCredits(const int32 credits) override;

        void TryDequeuePartition(int32 num_elements, OpKernelContext *ctx, CallbackWithResult callback) override;

        void TryDequeueMany(int32 num_elements, OpKernelContext *ctx, CallbackWithResult callback) override;
        void TryDequeue(OpKernelContext *ctx, CallbackWithResult callback) override;


      void TryDequeueForRequest(const IDandCountType &request_id, OpKernelContext *ctx,
                                CallbackWithPlainResult callback);

        Status MatchesNodeDef(const NodeDef &node_def) const override;

    private:
        mutex awaited_requests_lock_;
        std::unordered_set<IDandCountType> awaited_requests_ GUARDED_BY(awaited_requests_lock_);

        Status claim_id(const IDandCountType &idc);
        void release_id(const IDandCountType &idc);

      TF_DISALLOW_COPY_AND_ASSIGN(EgressGate);
    };

}  // namespace tensorflow
