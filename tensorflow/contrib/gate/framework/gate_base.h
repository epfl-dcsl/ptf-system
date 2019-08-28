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

#include <vector>
#include <deque>
#include <list>

#include <boost/optional.hpp>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "gate_interface.h"
#include "tensor_slicing.h"

namespace tensorflow {

// Functionality common to asynchronous BarrierInterface implementations.
class GateBase : public GateInterface {
 public:
  // Args:
  //   component_dtypes: The types of each component in a queue-element tuple.
  //   component_shapes: The shapes of each component in a queue-element tuple,
  //     which must either be empty (if the shapes are not specified) or
  //     or have the same size as component_dtypes.
  //   name: A name to use for the queue.
  GateBase(const DataTypeVector &component_dtypes, const std::vector<TensorShape> &component_shapes,
           const string &name, const int32 capacity, const bool limit_upstream,
           const bool limit_downstream);

  // Implementations of BarrierInterface methods --------------------------------
  const DataTypeVector& component_dtypes() const override;

  Status ValidateTuple(const Tuple& tuple) override;
  Status ValidateManyTuple(const Tuple& tuple) override;

    void Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
             DoneCallback callback) override;

  // Other public methods -----------------------------------------------------
  const std::vector<TensorShape>& component_shapes() const;

    Status MatchesNodeDef(const NodeDef &node_def) const override;

    Status SupplyDownstreamCredits(const int32 credits) override;

    void TryRequestUpstreamCredits(OpKernelContext *ctx, GateInterface::CallbackWithCredits &&callback) override;

    size_t OpenRequestCount() const override;

    size_t NumOpenRounds() const override;


 protected:
  enum class Action { kEnqueue, kDequeue };
  enum class RunResult {
      kProgressImpossible, // return when no other operation of the same type could make any progress (e.g. full buffer)
      kNoProgress, // return when this particular operation didn't succeed, but others of the same type could
      kComplete // return when this particular operation succeeds
  };

  // Tries to enqueue/dequeue (or close) based on whatever is at the
  // front of enqueue_attempts_/dequeue_attempts_.  Appends to
  // *finished the callback for any finished attempt (so it may be
  // called once mu_ is released).  Returns true if any progress was
  // made.
  struct CleanUp {
    CleanUp(DoneCallback&& f, CancellationToken ct, CancellationManager* cm)
        : finished(f), to_deregister(ct), cm(cm) {}
    DoneCallback finished;
    CancellationToken to_deregister;
    CancellationManager* cm;
  };

  // Returns the number of components in a queue-element tuple.
  int32 num_components() const;

  // Code common to Validate*Tuple().
  Status ValidateTupleCommon(const Tuple& tuple) const;

  TensorShape ManyOutShape(int i, int64 batch_size) const;

  void Cancel(Action action, CancellationManager* cancellation_manager,
              CancellationToken token);

  // Helper for cancelling all pending Enqueue(Many) operations when
  // Close is called with cancel_pending_enqueues.
  void CloseAndCancel();

  bool TryAttemptLocked(Action action, std::vector<CleanUp>* clean_up)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Tries to make progress on the enqueues or dequeues at the front
  // of the *_attempts_ queues.
  void FlushUnlocked();

  ~GateBase() override;

  // Helpers for implementing MatchesNodeDef().
  static string ShapeListString(const gtl::ArraySlice<TensorShape>& shapes);
  Status MatchesNodeDefOp(const NodeDef& node_def, const string& op) const;
  Status MatchesNodeDefTypes(const NodeDef& node_def) const;
  Status MatchesNodeDefShapes(const NodeDef& node_def) const;
  Status MatchesNodeDefCapacity(const NodeDef& node_def) const;
  Status MatchesNodeDefCreditLimitations(const NodeDef& node_def) const;
  Status MatchesNodeDefOpAndAttributes(const NodeDef& node_def, const string& op) const;
  Status MatchesNodeDefHelper(const NodeDef &node_def) const;

  const DataTypeVector component_dtypes_;
  const std::vector<TensorShape> component_shapes_;
  const std::string name_;
  const bool limit_requests_from_upstream_gate_, limit_requests_to_downstream_gate_;

  mutable mutex mu_;
  bool closed_ GUARDED_BY(mu_);

  struct Attempt;
  using RunCallback = std::function<RunResult(Attempt*)>;
  struct Attempt {
    int32 elements_requested; // TODO I think this is unneeded?
    DoneCallback done_callback;  // must be run outside mu_
    OpKernelContext* context;
    CancellationManager* cancellation_manager;  // not owned
    CancellationToken cancellation_token;
    RunCallback run_callback;  // must be run while holding mu_
    bool is_cancelled;
    Tuple tuple;

    Attempt(int32 elements_requested, DoneCallback done_callback,
            OpKernelContext* context, CancellationManager* cancellation_manager,
            CancellationToken cancellation_token, RunCallback run_callback);
  };

    class OpenRequest {
    public:
        OpenRequest(GateBase &base, const Tensor &id_and_count);

        // to know if we can use this to satisfy a partial request
        bool all_rounds_enqueued() const noexcept;

        // true if all_rounds_enqueued() and there is nothing left to dequeue
        // eligible for removal at this point
        bool exhausted() const noexcept;

        // returns the number of available rounds
        size_t available_rounds() const noexcept;

        size_t dequeued() const noexcept;
        size_t processed() const noexcept;

        //// Now the methods for the subclasses to call

        Status Enqueue(OpKernelContext *ctx, const Tuple &tuple) TF_MUST_USE_RESULT;
        Status EnqueueBatch(OpKernelContext *ctx, const Tuple &tuple) TF_MUST_USE_RESULT;

        Status Dequeue(OpKernelContext *ctx, GateInterface::Tuple &tuple) TF_MUST_USE_RESULT;
        Status DequeueMany(OpKernelContext *ctx, GateInterface::Tuple &tuple, int32 num_requested) TF_MUST_USE_RESULT;

        IDandCountType id() const noexcept;
        IDandCountType count() const noexcept;

        const Tensor *id_and_count(OpKernelContext *ctx);
        void set_id_and_count_batching(OpKernelContext *ctx, size_t batch_size);
    private:

        Status AllocateManyTupleShape(OpKernelContext *ctx, Tuple &tuple, const int32 batch_size, Round &example_round);

        IDandCountType id_, count_, dequeued_ = 0;

        PersistentTensor id_and_count_;
        std::deque<GateInterface::Round> available_rounds_;
        GateBase &gate_;
    };

  void
  TryEnqueueCommon(OpKernelContext *ctx, DoneCallback callback, int32 num_elements,
                   const string &op_type, RunCallback &&attempt);

    using DefaultRequest = std::function<Status(Attempt*, OpenRequest&)>;
    void
    TryEnqueueDefault(OpKernelContext *ctx, DoneCallback done_callback, int32 num_elements,
                      const string &op_type, const Tensor &id_and_count, DefaultRequest &&callback);

  void
  TryDequeueCommon(OpKernelContext *ctx, CallbackWithResult callback, int32 num_elements,
                   const string &op_type, RunCallback &&attempt);

    void
    TryDequeueDefault(OpKernelContext *ctx, int32 num_elements, const string &op_type,
                          CallbackWithResult result_callback, DefaultRequest &&callback);

    ///// All the stuff related to open rounds /////

    OpenRequest * FindOpenRequest(const Tensor &id_and_count);
    OpenRequest * FindOpenRequest(const IDandCountType &request_id);

    std::pair<Status,OpenRequest*> FindOrMakeNewRequest(const Tensor &id_and_count);

    // Returns true if this gate is closing, so that dequeue operations may empty out partial batches
    bool closing() const noexcept;

    // used for dequeuing
    bool has_open_requests() const noexcept;

    // Returns true if there are no more possible rounds for this gate to dequeue (closed && no partial batches)
    // if this is true, then all dequeue attempts should return OutOfRange
    bool exhausted() const noexcept;

    // returns the number of rounds that are opened
    size_t num_opened_requests() const noexcept;

    /*
     * Max open rounds is the maximum number allowed to be opened at any given time
     * opened_requests_ is the amount given out already. We need this to be separate from the number of requests
     * actually in this gate (e.g. looking at the sizes of the deques that hold the requests) because it tracks
     * the number of requests / credits GIVEN OUT by this gate.
     *
     * Note that the size of request_map_ can't be used because otherwise the propagation will never stop unless the request map fills up
     * the size of max_open_requests_, which has a low probability of it defaults to max.
     */
    const int64 max_open_requests_;
    int64 opened_requests_ = 0 GUARDED_BY(mu_); // we need this to track how many credits we have already given out to the upstream

    // protected because EgressGate needs to use it too
    void CloseRequest(OpenRequest *open_req);

    // string LogGateState() const noexcept;

private:

    /*
     * Ancillary data structure to enable fast enqueue of rounds
     * This dataset is as large as open_rounds_ at all times
     */
    std::unordered_map<IDandCountType,OpenRequest> request_map_ GUARDED_BY(mu_);

    /*
     * An ordering of items in the round_map_ based on their arrival order.
     *
     * This data structure is exactly the size of open_rounds_
     *
     * This is a vector because shuffling is easy
     */
    std::vector<OpenRequest*> ordered_unopen_requests_, ordered_open_requests_ GUARDED_BY(mu_);

    bool OpenNewRequest(OpenRequest **open_req);

    bool OpenRequestSatisfiesDemandPossiblyAdjusted(OpenRequest &open_req, Attempt &attempt) const;

    std::list<Attempt> enqueue_attempts_, dequeue_attempts_ GUARDED_BY(mu_);

    // DOWNSTREAM: Used for determining the number of slots to give out
    int64 available_downstream_requests_ = 0 GUARDED_BY(mu_);

    class CreditRequest {
    public:
        CreditRequest(OpKernelContext *ctx, CallbackWithCredits &&callback, CancellationManager *cm, CancellationToken ct);
        CreditRequest &operator=(CreditRequest &&) = default;
        CreditRequest(CreditRequest &&) = default;

        bool matches_cancellation(CancellationManager *cm, CancellationToken ct) const noexcept;
        CallbackWithCredits& get_callback();
        void cancel(const std::string &gate_name);
        void add_to_cleanup(std::vector<CleanUp> &cleanup, int32 available_credits) const;
        void add_error_to_cleanup(std::vector<CleanUp> &cleanup) const;
    private:

        CancellationToken ct_;
        CancellationManager * cm_;
        CallbackWithCredits callback_;
        OpKernelContext * ctx_;
        TF_DISALLOW_COPY_AND_ASSIGN(CreditRequest);
    };

    boost::optional<CreditRequest> credit_request_;

    void CancelCreditRequest(CancellationManager *cm, CancellationToken ct);

    void TryCreditRequestLocked(std::vector<CleanUp> &cleanup);

    std::string LogGateState() const noexcept;

public:
  TF_DISALLOW_COPY_AND_ASSIGN(GateBase);
};

}  // namespace tensorflow
