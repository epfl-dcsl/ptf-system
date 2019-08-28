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

#include <string>
#include <vector>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// All implementations must be thread-safe.
class GateInterface : public ResourceBase {
 public:
  using Round = std::vector<PersistentTensor>;
  using Tuple = std::vector<Tensor>;
  using IDandCountType = int32;
  using DoneCallback = AsyncOpKernel::DoneCallback;
  using CallbackWithResult = std::function<void(const Tensor &id_and_count, const Tuple &data)>;

  virtual Status ValidateTuple(const Tuple& tuple) = 0;
  virtual Status ValidateManyTuple(const Tuple& tuple) = 0;

  // Stashes a function object for future execution, that will eventually
  // enqueue the tuple of tensors into the queue, and returns immediately. The
  // function object is guaranteed to call 'callback'.
  virtual void TryEnqueue(const Tensor &id_and_count, const Tuple &tuple, OpKernelContext *ctx, DoneCallback callback) = 0;

    virtual void TryEnqueueMany(const Tensor &id_and_count, const Tuple &tuple, OpKernelContext *ctx, DoneCallback callback) = 0;

  // Stashes a function object for future execution, that will eventually
  // dequeue an element from the queue and call 'callback' with that tuple
  // element as argument.
  virtual void TryDequeue(OpKernelContext* ctx, CallbackWithResult callback) = 0;

  virtual void TryDequeueMany(int32 num_elements, OpKernelContext *ctx, CallbackWithResult callback) = 0;

  virtual void TryDequeuePartition(int32 num_elements, OpKernelContext *ctx, CallbackWithResult callback) = 0;

  // Signals that no more elements will be enqueued, and optionally
  // cancels pending Enqueue(Many) operations.
  //
  // After calling this function, subsequent calls to Enqueue(Many)
  // will fail. If `cancel_pending_enqueues` is true, all pending
  // calls to Enqueue(Many) will fail as well.
  //
  // After calling this function, all current and subsequent calls to
  // Dequeue(Many) will fail instead of blocking (though they may
  // succeed if they can be satisfied by the elements in the queue at
  // the time it was closed).
  virtual void Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
                     DoneCallback callback) = 0;

    // Assuming *this represents a shared queue, verify that it matches
  // another instantiation indicated by node_def.
  virtual Status MatchesNodeDef(const NodeDef& node_def) const = 0;

  virtual const DataTypeVector& component_dtypes() const = 0;

  string DebugString() override;

  static IDandCountType get_id(const Tensor &id_and_count);
  static IDandCountType get_count(const Tensor &id_and_count);

  /// Credit-based methods ///

    /*
     * Supplies a strictly >0 number of credits to this gate.
     */
  virtual Status SupplyDownstreamCredits(const int32 credits) = 0;

    /*
     * Blocks until at least one credit is available, or there is an error
     *
     * If Status is not ok upon return, the value in the second part is undefined.
     * If status is ok, then the second number will be at least 1
     */
  using CallbackWithCredits = std::function<void(int32)>;
  virtual void TryRequestUpstreamCredits(OpKernelContext *ctx, CallbackWithCredits &&callback) = 0;

    virtual size_t OpenRequestCount() const = 0;
    virtual size_t NumOpenRounds() const = 0;

protected:
  virtual ~GateInterface() {}
};

}  // namespace tensorflow
