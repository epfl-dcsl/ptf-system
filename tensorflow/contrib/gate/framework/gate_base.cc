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
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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


#include <vector>
#include <algorithm>
#include <sstream>
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "gate_base.h"
#include "tensor_slicing.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  const DataTypeVector& GateBase::component_dtypes() const {
    return component_dtypes_;
  }

  const std::vector<TensorShape>& GateBase::component_shapes() const {
    return component_shapes_;
  }

    int32 GateBase::num_components() const { return component_dtypes_.size(); }

  TensorShape GateBase::ManyOutShape(int i, int64 batch_size) const {
    TensorShape shape({batch_size});
    shape.AppendShape(component_shapes_[i]);
    return shape;
  }

GateBase::GateBase(const DataTypeVector &component_dtypes, const std::vector<TensorShape> &component_shapes,
                   const string &name, const int32 capacity, const bool limit_upstream,
                   const bool limit_downstream)
    : component_dtypes_(component_dtypes),
      component_shapes_(component_shapes),
      name_(name),
      closed_(false),
      max_open_requests_(capacity),
      limit_requests_to_downstream_gate_(limit_downstream),
      limit_requests_from_upstream_gate_(limit_upstream) {
  auto components_size = component_shapes.size();
  auto dtypes_size = component_dtypes.size();
  if (dtypes_size == 0) {
    LOG(ERROR) << "Gate " << name << " has no specified dtypes!";
    closed_ = true; return;
  } else if (dtypes_size != components_size) {
    LOG(ERROR) << "Gate dtype count is " << dtypes_size << " while shape count is " << components_size << ". These must be equal!";
    closed_ = true; return;
  } else if (capacity < 1) {
    LOG(ERROR) << "Gate capacity is <1: " << capacity;
    closed_ = true; return;
  }
}

GateBase::~GateBase() {}

Status GateBase::ValidateTupleCommon(const Tuple& tuple) const {
  if (tuple.size() != static_cast<size_t>(num_components())) {
    return errors::InvalidArgument(
        "Wrong number of components in tuple. Expected ", num_components(),
        ", got ", tuple.size());
  }
  for (size_t i = 0; i < tuple.size(); ++i) {
    if (tuple[i].dtype() != component_dtypes_[i]) {
      return errors::InvalidArgument(
          "Type mismatch in tuple component ", i, ". Expected ",
          DataTypeString(component_dtypes_[i]), ", got ",
          DataTypeString(tuple[i].dtype()));
    }
  }
  return Status::OK();
}

// static
string GateBase::ShapeListString(const gtl::ArraySlice<TensorShape>& shapes) {
  string result = "[";
  bool first = true;
  for (const TensorShape& shape : shapes) {
    strings::StrAppend(&result, (first ? "" : ", "), shape.DebugString());
    first = false;
  }
  strings::StrAppend(&result, "]");
  return result;
}

  Status GateBase::MatchesNodeDef(const NodeDef &node_def) const {
      return MatchesNodeDefHelper(node_def);
  }

    Status GateBase::MatchesNodeDefHelper(const NodeDef &node_def) const {
      TF_RETURN_IF_ERROR(MatchesNodeDefShapes(node_def));
      TF_RETURN_IF_ERROR(MatchesNodeDefTypes(node_def));
      TF_RETURN_IF_ERROR(MatchesNodeDefCapacity(node_def));
      TF_RETURN_IF_ERROR(MatchesNodeDefCreditLimitations(node_def));
      return Status::OK();
    }


Status GateBase::MatchesNodeDefOp(const NodeDef& node_def,
                                     const string& op) const {
  if (node_def.op() != op) {
    return errors::InvalidArgument("Shared gate '", name_, "' has type '", op,
                                   "' that does not match type of Node '",
                                   node_def.name(), "': ", node_def.op());
  }
  return Status::OK();
}

Status GateBase::MatchesNodeDefOpAndAttributes(const NodeDef &node_def, const string &op) const {
  TF_RETURN_IF_ERROR(MatchesNodeDefOp(node_def, op));
  TF_RETURN_IF_ERROR(MatchesNodeDefHelper(node_def));
  return Status::OK();
}

Status GateBase::MatchesNodeDefTypes(const NodeDef& node_def) const {
  DataTypeVector requested_dtypes;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(node_def, "component_types", &requested_dtypes));
  if (requested_dtypes != component_dtypes_) {
    return errors::InvalidArgument("Shared gate '", name_,
                                   "' has component types ",
                                   DataTypeSliceString(component_dtypes_),
                                   " but requested component types were ",
                                   DataTypeSliceString(requested_dtypes));
  }
  return Status::OK();
}

Status GateBase::MatchesNodeDefShapes(const NodeDef& node_def) const {
  std::vector<TensorShape> requested_shapes;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "component_shapes", &requested_shapes));
  if (requested_shapes != component_shapes_) {
    return errors::InvalidArgument("Shared gate '", name_,
                                   "' has component shapes ",
                                   ShapeListString(component_shapes_),
                                   " but requested component shapes were ",
                                   ShapeListString(requested_shapes));
  }
  return Status::OK();
}

Status GateBase::MatchesNodeDefCapacity(const NodeDef& node_def) const {
  int32 capacity;
  TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "capacity", &capacity));
  if (capacity != max_open_requests_) {
    return errors::InvalidArgument("Shared gate '", name_,
                                   "' has capacity ",
                                   max_open_requests_,
                                   " but requested capacity was ",
                                   capacity
    );
  }
  return Status::OK();

}
    Status GateBase::MatchesNodeDefCreditLimitations(const NodeDef& node_def) const {
      bool limit_upstream, limit_downstream;
      if (HasNodeAttr(node_def, "limit_upstream")) {
        TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "limit_upstream", &limit_upstream));
        if (limit_upstream != limit_requests_from_upstream_gate_) {
          return InvalidArgument("Shared gate '", name_, "' is ", (limit_requests_from_upstream_gate_ ? "true" : "false"), " for limiting upstream credits, but requested limit was ",
                                 (limit_upstream ? "true" : "false"));
        }
      }
      if (HasNodeAttr(node_def, "limit_downstream")) {
        TF_RETURN_IF_ERROR(GetNodeAttr(node_def, "limit_downstream", &limit_downstream));
        if (limit_downstream != limit_requests_to_downstream_gate_) {
          return InvalidArgument("Shared gate '", name_, "' is ", (limit_requests_to_downstream_gate_ ? "true" : "false"), " for limiting downstream credits, but requested limit was ",
                                 (limit_downstream ? "true" : "false"));
        }
      }
      return Status::OK();
    }

Status GateBase::ValidateTuple(const Tuple& tuple) {
  TF_RETURN_IF_ERROR(ValidateTupleCommon(tuple));
  for (size_t i = 0; i < tuple.size(); ++i) {
      auto const & t = tuple[i];
      auto const & shape = t.shape();
      auto dtype = t.dtype();
    if (!component_shapes_[i].IsSameSize(shape)) {
      return errors::InvalidArgument(
              "Shape mismatch in tuple component ", i, ". Expected ",
              component_shapes_[i].DebugString(), ", got ",
              shape.DebugString());
    }
    if (dtype != component_dtypes_[i]) {
      return errors::InvalidArgument("Dtype mismatch in tuple component ", i,
      ". Expected ", component_dtypes_[i], ", but got ", dtype);
    }
  }
  return Status::OK();
}

Status GateBase::ValidateManyTuple(const Tuple& tuple) {
  TF_RETURN_IF_ERROR(ValidateTupleCommon(tuple));
  const int64 batch_size = tuple[0].dim_size(0);
   if (batch_size == 0) {
     return InvalidArgument("First stride dim is 0. Must be strictly positive");
   }
  for (size_t i = 0; i < tuple.size(); ++i) {
    auto const & t = tuple[i];
    auto const & shape = t.shape();
    auto dtype = t.dtype();
    // Expected shape is [batch_size] + component_shapes_[i]
    const TensorShape expected_shape = ManyOutShape(i, batch_size);
    if (!expected_shape.IsSameSize(shape)) {
      return errors::InvalidArgument("ManyTuple Shape mismatch in tuple component ", i,
                                     ". Expected ",
                                     expected_shape.DebugString(), ", got ",
                                     shape.DebugString());
    }
    if (dtype != component_dtypes_[i]) {
      return errors::InvalidArgument("Dtype mismatch in tuple component ", i,
                                     ". Expected ", component_dtypes_[i], ", but got ", dtype);
    }
  }
  return Status::OK();
}

void GateBase::Cancel(Action action, CancellationManager* cancellation_manager,
                       CancellationToken token) {
  DoneCallback callback = nullptr;
  {
    mutex_lock lock(mu_);
    auto attempts =
        action == Action::kEnqueue ? &enqueue_attempts_ : &dequeue_attempts_;

    for (Attempt& attempt : *attempts) {
      if (attempt.cancellation_manager == cancellation_manager &&
          attempt.cancellation_token == token) {
        if (!attempt.is_cancelled) {
          attempt.is_cancelled = true;
          if (action == Action::kEnqueue) {
            attempt.context->SetStatus(
                errors::Cancelled("Enqueue operation was cancelled"));
          } else {
            attempt.context->SetStatus(
                errors::Cancelled("Dequeue operation was cancelled"));
          }
          std::swap(callback, attempt.done_callback);
        }
        break;
      }
    }
  }
  if (callback) {
    callback();
    FlushUnlocked();
  }
}

void GateBase::CloseAndCancel() {
  std::vector<DoneCallback> callbacks;
  {
    mutex_lock lock(mu_);
    closed_ = true;
    for (Attempt& attempt : enqueue_attempts_) {
      if (!attempt.is_cancelled) {
        attempt.is_cancelled = true;
        attempt.context->SetStatus(
            errors::Cancelled("Enqueue operation was cancelled"));
        callbacks.emplace_back(std::move(attempt.done_callback));
      }
    }
  }
  for (const DoneCallback& callback : callbacks) {
    callback();
  }
  FlushUnlocked();
}

void GateBase::Close(OpKernelContext* ctx, bool cancel_pending_enqueues,
                      DoneCallback callback) {
  if (cancel_pending_enqueues) {
    CloseAndCancel();
    callback();
  } else {
    {
      mutex_lock lock(mu_);
      enqueue_attempts_.emplace_back(
          0, callback, ctx, nullptr, CancellationManager::kInvalidToken,
          [this](Attempt* attempt) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            if (closed_) {
              attempt->context->SetStatus(
                  errors::Cancelled("Gate '", name_, "' is already closed."));
            } else {
              closed_ = true;
            }
            return RunResult::kComplete;
          });
    }
    FlushUnlocked();
  }
}

bool GateBase::TryAttemptLocked(Action action,
                                   std::vector<CleanUp>* clean_up) {
  auto attempts = action == Action::kEnqueue ? &enqueue_attempts_ : &dequeue_attempts_;

  bool progress = false;
  bool done = false;
  for (auto it = attempts->begin(); it != attempts->end() and not done;) {
      if (it->is_cancelled) {
        if (action == Action::kEnqueue) {
          if (closed_) {
            VLOG(1) << "Skipping cancelled enqueue attempt";
          } else {
            LOG(WARNING)
                    << name_
                    << ": Skipping cancelled enqueue attempt with queue not closed";
          }
        } else {
          if (closed_) {
            VLOG(1) << "Skipping cancelled dequeue attempt";
          } else {
            LOG(WARNING)
                    << name_
                    << ": Skipping cancelled dequeue attempt with queue not closed";
          }
        }
        it = attempts->erase(it);
      } else {
        Attempt* cur_attempt = &attempts->front();
        switch (cur_attempt->run_callback(cur_attempt)) {
          case RunResult::kProgressImpossible: // attempt couldn't proceed. abort!
              // no need to advance it++. for loop will exit
            done = true;
            break;
          case RunResult::kNoProgress:
            it++;
            break;
          case RunResult::kComplete: // dequeue attempt was processed successfully. we can attempt another one
            progress = true;
            clean_up->emplace_back(std::move(cur_attempt->done_callback),
                                   cur_attempt->cancellation_token,
                                   cur_attempt->context->cancellation_manager());
            it = attempts->erase(it);
            break;
        }
      }
  }
  return progress;
}

void GateBase::FlushUnlocked() {
  std::vector<CleanUp> clean_up;
  Ref();
  {
    mutex_lock lock(mu_);
    bool changed;
    do {
      changed = TryAttemptLocked(Action::kEnqueue, &clean_up);
      changed = TryAttemptLocked(Action::kDequeue, &clean_up) or changed; // this order so dequeue attempts are tried
    } while (changed);
    TryCreditRequestLocked(clean_up);
  }
  Unref();
  for (const auto& to_clean : clean_up) {
    if (to_clean.to_deregister != CancellationManager::kInvalidToken) {
      // NOTE: We can safely ignore the return value of
      // DeregisterCallback because the mutex mu_ ensures that the
      // cleanup action only executes once.
      to_clean.cm->DeregisterCallback(to_clean.to_deregister);
    }
    to_clean.finished();
  }
}

  GateBase::Attempt::Attempt(int32 elements_requested, GateInterface::DoneCallback done_callback,
                                OpKernelContext *context, CancellationManager *cancellation_manager,
                                CancellationToken cancellation_token, GateBase::RunCallback run_callback)
          : elements_requested(elements_requested),
            done_callback(done_callback),
            context(context),
            cancellation_manager(cancellation_manager),
            cancellation_token(cancellation_token),
            run_callback(run_callback),
            is_cancelled(false) {}

  void GateBase::TryEnqueueCommon(OpKernelContext *ctx, GateInterface::DoneCallback callback, int32 num_elements,
                                     const string &op_type, GateBase::RunCallback &&attempt) {
    CancellationManager *cm = ctx->cancellation_manager();
    CancellationToken token = cm->get_cancellation_token();
    bool already_cancelled;
    {
      mutex_lock l(mu_);
      already_cancelled = !cm->RegisterCallback(
              token, [this, cm, token]() { Cancel(Action::kEnqueue, cm, token); });
      if (!already_cancelled) {
        enqueue_attempts_.emplace_back(
                num_elements, callback, ctx, cm, token, attempt
        );
      }
    }
    if (!already_cancelled) {
      FlushUnlocked();
    } else {
      ctx->SetStatus(errors::Cancelled(op_type, " operation was cancelled"));
      callback();
    }
  }

    void GateBase::TryEnqueueDefault(OpKernelContext *ctx, GateInterface::DoneCallback done_callback, int32 num_elements,
                                     const string &op_type, const Tensor &idc,
                                     DefaultRequest &&callback) {
      TryEnqueueCommon(ctx, done_callback, num_elements, op_type, [this, idc, callback](Attempt *attempt) {
          auto *const context = attempt->context;
          if (closed_) {
            context->SetStatus(Cancelled(name_, " is closed."));
          } else {
            auto status_pair = FindOrMakeNewRequest(idc);
            if (not status_pair.first.ok()) {
              auto &status = status_pair.first;
              if (IsOutOfRange(status)) {
                return RunResult::kNoProgress;
              } else {
                context->SetStatus(status_pair.first);
                return RunResult::kComplete;
              }
            }
            auto open_request_p = status_pair.second;
            DCHECK_NE(open_request_p, nullptr);

            auto status = callback(attempt, *open_request_p);
            if (not status.ok()) {
              context->SetStatus(status);
            }
          }
         return RunResult::kComplete;
      });
    }

  void GateBase::TryDequeueCommon(OpKernelContext *ctx, CallbackWithResult callback,
                                     int32 num_elements, const string &op_type,
                                     RunCallback &&attempt){
    if (num_elements == 0) {
      ctx->SetStatus(errors::Internal("Attempting to dequeue 0 elements for op type ", op_type));
      callback(Tensor(), Tuple());
      return;
    }

    CancellationManager* cm = ctx->cancellation_manager();
    CancellationToken token = cm->get_cancellation_token();
    bool already_cancelled;
    {
      mutex_lock l(mu_);

      already_cancelled = !cm->RegisterCallback(
              token, [this, cm, token]() { Cancel(Action::kDequeue, cm, token); });
      if (!already_cancelled) {
        dequeue_attempts_.emplace_back(num_elements, [callback]() { callback(Tensor(), Tuple()); }, ctx, cm, token, attempt);
      }
    }
    if (!already_cancelled) {
      FlushUnlocked();
    } else {
      ctx->SetStatus(errors::Cancelled("Dequeue operation was cancelled"));
      callback(Tensor(), Tuple());
    }
  }

    void
    GateBase::TryDequeueDefault(OpKernelContext *ctx, int32 num_elements, const string &op_type,
                                    CallbackWithResult result_callback, DefaultRequest &&callback) {
      TryDequeueCommon(ctx, result_callback, num_elements, op_type, [this, callback](Attempt *attempt) {
          auto * const context = attempt->context;

          if (exhausted()) {
            context->SetStatus(OutOfRange(name_, " has no more elements."));
            return RunResult::kComplete;
          }

          decltype(ordered_open_requests_) to_delete;
          auto result = RunResult::kProgressImpossible;

          // First, we attempt all the possible open requests
          if (has_open_requests()) {
            for (auto &open_request_p : ordered_open_requests_) {
              auto &open_request = *open_request_p;
              if (OpenRequestSatisfiesDemandPossiblyAdjusted(open_request, *attempt)) {
                DCHECK_GT(open_request.available_rounds(), 0);
                auto status = callback(attempt, open_request);
                if (not status.ok()) {
                  context->SetStatus(status);
                }
                if (open_request.exhausted()) {
                  to_delete.emplace_back(open_request_p);
                }
                result = RunResult::kComplete;
                break;
              } else if (closing() and open_request.exhausted()) {
                LOG(WARNING) << name_ << " TryDequeueDefault pruning empty open request on closing() condition. ID: " << open_request.id();
                to_delete.emplace_back(open_request_p);
              }
            }
          }

          /*
           * We don't need to check the closing() and is exhausted condition like with the opened requests because
           * by construction, any open request must have SOME dequeuable rounds.
           */
          OpenRequest *open_req_p;
          while (result == RunResult::kProgressImpossible and OpenNewRequest(&open_req_p)) {
            auto &open_req = *open_req_p;
            DCHECK_GT(open_req.available_rounds(), 0);
            if (OpenRequestSatisfiesDemandPossiblyAdjusted(open_req, *attempt)) {
              auto status = callback(attempt, open_req);
              if (not status.ok()) {
                context->SetStatus(status);
              }
              if (open_req.exhausted()) {
                to_delete.emplace_back(open_req_p);
              }
              result = RunResult::kComplete;
            }
          }
          for (auto &to_delete_p : to_delete) {
            CloseRequest(to_delete_p);
          }

          return result;
      });
    }

    bool GateBase::OpenNewRequest(GateBase::OpenRequest **open_req) {
      if ((not limit_requests_to_downstream_gate_ or available_downstream_requests_ > 0) and not ordered_unopen_requests_.empty()) {
        auto new_request = ordered_unopen_requests_.front();
        ordered_open_requests_.push_back(new_request);
        ordered_unopen_requests_.erase(ordered_unopen_requests_.begin());
        available_downstream_requests_ = max(available_downstream_requests_-1,
                                           static_cast<decltype(available_downstream_requests_)>(0));
        *open_req = new_request;
        return true;
      }
      return false;
    }

    bool GateBase::OpenRequestSatisfiesDemandPossiblyAdjusted(GateBase::OpenRequest &open_req, GateBase::Attempt &attempt) const {
      auto available_rounds = open_req.available_rounds();
      auto requested_rounds = attempt.elements_requested;
      if (available_rounds >= requested_rounds or open_req.all_rounds_enqueued()) {
        return true;
      } else if (closing() and available_rounds > 0 and available_rounds < requested_rounds) {
        attempt.elements_requested = available_rounds;
        return true;
      }
      return false;
    }

    bool GateBase::closing() const noexcept {
        return closed_ && enqueue_attempts_.empty();
    }

    bool GateBase::has_open_requests() const noexcept {
      return not ordered_open_requests_.empty();
    }

    bool GateBase::exhausted() const noexcept {
      return closed_ and request_map_.empty();
    }

    Status GateBase::SupplyDownstreamCredits(const int32 credits) {
      {
        mutex_lock l(mu_);
        if (credits < 1) {
          return Internal(name_, " got a non-positive number of credits to supply for downstream: ", credits);
        } else if (exhausted()) {
          return Cancelled(name_, " is closed and exhausted. No more credits needed.");
        }
        available_downstream_requests_ += credits;
        DCHECK_LE(available_downstream_requests_, kint32max);
      }
      FlushUnlocked();
      return Status::OK();
    }

    void GateBase::TryRequestUpstreamCredits(OpKernelContext *ctx, GateInterface::CallbackWithCredits &&callback) {
      CancellationManager *cm = ctx->cancellation_manager();
      CancellationToken token = cm->get_cancellation_token();
      bool already_cancelled = true; // set to true for the is_initialized case
      {
        mutex_lock l(mu_);
        if (closed_) {
          ctx->SetStatus(errors::Cancelled(name_, " is closed. Can't set a credit request"));
          return;
        }

        if (credit_request_.is_initialized()) {
          ctx->SetStatus(errors::Aborted(name_, " attempting to set >1 simultaneous credit request"));
        } else {
          already_cancelled = not cm->RegisterCallback(token, [this, cm, token]() { CancelCreditRequest(cm, token); });
          if (not already_cancelled) {
            credit_request_ = CreditRequest(ctx, move(callback), cm, token);
          }
        }
      }
      if (already_cancelled) {
        ctx->SetStatus(errors::Cancelled(name_, " request for credits cancelled."));
        callback(-1);
      } else {
        FlushUnlocked();
      }
    }

    void GateBase::CloseRequest(GateBase::OpenRequest *open_req) {
      auto id = open_req->id();
      auto ret = request_map_.erase(id);
      DCHECK_EQ(ret, 1);
      bool found_request = false;
      auto found = find(
              ordered_open_requests_.begin(),
              ordered_open_requests_.end(), open_req
      );
      if (found != ordered_open_requests_.end()) {
        ordered_open_requests_.erase(found);
          found_request = true;
      }
      found = find(
              ordered_unopen_requests_.begin(),
              ordered_unopen_requests_.end(), open_req
      );
      if (found != ordered_unopen_requests_.end()) {
        ordered_unopen_requests_.erase(found);
        found_request = true;
      }
      DCHECK(found_request);
      if (limit_requests_from_upstream_gate_) {
        opened_requests_ -= 1;
      }
    }

    GateBase::OpenRequest * GateBase::FindOpenRequest(const Tensor &id_and_count) {
      auto id = get_id(id_and_count);
        return FindOpenRequest(id);
    }

    GateBase::OpenRequest *GateBase::FindOpenRequest(const GateInterface::IDandCountType &request_id) {
      auto count = request_map_.count(request_id);
      if (count > 0) {
        DCHECK_EQ(count, 1);
        return &request_map_.at(request_id);
      }
      return nullptr;
    }

    std::pair<Status, GateBase::OpenRequest *> GateBase::FindOrMakeNewRequest(const Tensor &id_and_count) {
      auto ptr = FindOpenRequest(id_and_count);
      if (ptr != nullptr) {
        return make_pair(Status::OK(), ptr);
      }
      auto num_existing_rounds = num_opened_requests();
      if (num_existing_rounds == max_open_requests_) {
        if (limit_requests_from_upstream_gate_) {
          return make_pair(Internal(name_, " is having its max_open_request (", max_open_requests_, ") limit exceeded!"), nullptr);
        } else {
          return make_pair(errors::OutOfRange(name_, " has maximum number of requests open (", max_open_requests_, "). Blocking until some close."), nullptr);
        }
      }
      auto id = get_id(id_and_count);
      auto result = request_map_.emplace(id, OpenRequest(*this, id_and_count));
      DCHECK(result.second);
      OpenRequest *or_value = &result.first->second, *open_req_unused;

      // Optimization: try to open as many as possible if we have credits
      // First, open existing ones with the while loop, then place the current one
      // in *open* or *unopen* requests if there are things available
      while (OpenNewRequest(&open_req_unused));
      if (not limit_requests_to_downstream_gate_ or available_downstream_requests_ > 0) {
        ordered_open_requests_.emplace_back(or_value);
        // I don't know why it doesn't just upcast 0!
        available_downstream_requests_ = max(available_downstream_requests_-1,
                                             static_cast<decltype(available_downstream_requests_)>(0));
      } else {
        ordered_unopen_requests_.emplace_back(or_value);
      }
      return make_pair(Status::OK(), or_value);
    }

    size_t GateBase::num_opened_requests() const noexcept {
      return request_map_.size();
    }

    void GateBase::CancelCreditRequest(CancellationManager *cm, CancellationToken ct) {
      // NOTE: do not call into the cancellation manager! this will only be called by the cancellation manager
      CallbackWithCredits callback = nullptr;
      {
        mutex_lock l(mu_);
        if (credit_request_.is_initialized()) {
          auto& credit_req = credit_request_.get();
          if (credit_req.matches_cancellation(cm, ct)) {
            credit_req.cancel(name_);
            swap(callback, credit_req.get_callback());
            credit_request_.reset();
          }
        }
      }
      if (callback) {
        callback(-1);
      }
    }

    void GateBase::TryCreditRequestLocked(std::vector<CleanUp> &cleanup) {
      if (credit_request_.is_initialized()) {
        auto &cr = credit_request_.get();
        if (closed_) {
          cr.cancel(name_);
          cr.add_error_to_cleanup(cleanup);
          credit_request_.reset();
        } else if (opened_requests_ < max_open_requests_) {
          auto available = max_open_requests_ - opened_requests_;
          DCHECK_GT(available, 0);
          DCHECK_LE(available, kint32max);
          cr.add_to_cleanup(cleanup, available);
          credit_request_.reset();
          opened_requests_ = max_open_requests_;
        }
      }
    }

    GateBase::OpenRequest::OpenRequest(GateBase &base, const Tensor &id_and_count) :
            gate_(base), id_and_count_(id_and_count),
            id_(get_id(id_and_count)),
            count_(get_count(id_and_count)) {}

    bool GateBase::OpenRequest::all_rounds_enqueued() const noexcept {
      return count_ == (dequeued_ + available_rounds());
    }

    bool GateBase::OpenRequest::exhausted() const noexcept {
        return dequeued_ == count_ and available_rounds() == 0;
    }

    size_t GateBase::OpenRequest::available_rounds() const noexcept {
        return available_rounds_.size();
    }

    Status GateBase::OpenRequest::Enqueue(OpKernelContext *ctx, const Tuple &tuple) {
        Round round;
        auto tuple_size = tuple.size();
        round.resize(tuple_size);
        for (decltype(tuple_size) i = 0; i < tuple_size; ++i) {
            Tensor *ptt;
            auto &pt = round[i];
            TF_RETURN_IF_ERROR(ctx->allocate_persistent(
                    gate_.component_dtypes_[i],
                    gate_.component_shapes_[i],
                    &pt,  &ptt
            ));
            *ptt = tuple[i];
        }
        available_rounds_.push_back(move(round));
        return Status::OK();
    }

    Status GateBase::OpenRequest::EnqueueBatch(OpKernelContext *ctx, const Tuple &tuple) {
      auto batch_size = tuple.at(0).dim_size(0);
      DCHECK_LT(processed(), count_);
      auto expected_remaining = count_ - processed();
      if (batch_size > expected_remaining) {
        return Internal("Incorrect EnqueueBatch! Expected ", expected_remaining, " remaining rounds, but got ", batch_size, ". Id_and_count: ", id_and_count_.AccessTensor(ctx)->DebugString());
      }

      auto tuple_size = tuple.size();

      for (decltype(batch_size) round = 0; round < batch_size; ++round) {
          Round open_round;
        open_round.resize(tuple_size);

        for (decltype(tuple_size) batch_component_index = 0; batch_component_index < tuple_size; ++batch_component_index) {
          Tensor *element_t;
          auto &element = open_round[batch_component_index];
            TF_RETURN_IF_ERROR(ctx->allocate_persistent(
                    gate_.component_dtypes_[batch_component_index],
                    gate_.component_shapes_[batch_component_index],
                    &element, &element_t
            ));
          auto &batch_component = tuple.at(batch_component_index);
          TF_RETURN_IF_ERROR(CopySliceToElement(batch_component, element_t, round));
        }

        available_rounds_.push_back(move(open_round));
      }

      return Status::OK();
    }

    Status GateBase::OpenRequest::Dequeue(OpKernelContext *ctx, GateInterface::Tuple &tuple) {
      if (available_rounds() == 0) {
        return ResourceExhausted("OpenRound is exhausted");
      }
      if (!tuple.empty()) {
        return Internal("Dequeue got a non-empty tuple");
      }
      auto &round = available_rounds_.front();
      for (auto &round_element : round) {
        auto elem = round_element.AccessTensor(ctx);
        tuple.push_back(*elem);
      }
      available_rounds_.pop_front();
      dequeued_++;
      DCHECK_LE(dequeued_, count_);
      return Status::OK();
    }

    Status GateBase::OpenRequest::DequeueMany(OpKernelContext *ctx, GateInterface::Tuple &tuple, int32 num_requested) {
      auto avail_rounds = available_rounds();
      if (avail_rounds == 0) {
        return Internal("OpenRound is exhausted. Caller needs to check preconditions better!");
      }
      if (avail_rounds < num_requested) {
        if (all_rounds_enqueued()) {
          num_requested = avail_rounds;
        } else {
          auto remaining_rounds = count_ - (dequeued_ + avail_rounds);
          return Internal(gate_.name_, " DequeueMany for id ", id_, ": need to wait for all rounds before partial dequeuing. Remaining rounds ", remaining_rounds,
                          ". Gate is ", (gate_.closing() ? "" : "not "), "closing");
        }
      }
      if (not tuple.empty()) {
        return Internal("DequeueMany got a non-empty tuple");
      }

      TF_RETURN_IF_ERROR(AllocateManyTupleShape(ctx, tuple, num_requested, available_rounds_.front()));

      for (int32 i = 0; i < num_requested; i++) {
        auto &round = available_rounds_.front();
        for (size_t elem_index = 0; elem_index < round.size(); elem_index++) {
          auto &round_elem = round[elem_index];
          auto elem = round_elem.AccessTensor(ctx);
          auto &dest_elem = tuple[elem_index];
          TF_RETURN_IF_ERROR(CopyElementToSlice(*elem, &dest_elem, i));
        }
        available_rounds_.pop_front();
      }
      dequeued_ += num_requested;
      DCHECK_LE(dequeued_, count_);
      return Status::OK();
    }

    Status
    GateBase::OpenRequest::AllocateManyTupleShape(OpKernelContext *ctx, GateInterface::Tuple &tuple,
                                                const int32 batch_size,
                                                Round &example_round) {
      DCHECK(tuple.empty());
      DCHECK(!example_round.empty());
      auto num_elems = example_round.size();
      tuple.resize(num_elems);

      for (size_t i = 0; i < num_elems; ++i) {
        auto &round_elem = example_round[i];
        auto &dest_elem = tuple[i];
        auto const elem = round_elem.AccessTensor(ctx);

        TensorShape shape(elem->shape());
        shape.InsertDim(0, batch_size);
        auto dtype = elem->dtype();
        TF_RETURN_IF_ERROR(ctx->allocate_temp(dtype, shape, &dest_elem));
      }
        return Status::OK();
    }

    GateInterface::IDandCountType GateBase::OpenRequest::id() const noexcept {
      return id_;
    }

    GateInterface::IDandCountType GateBase::OpenRequest::count() const noexcept {
      return count_;
    }

    size_t GateBase::OpenRequest::dequeued() const noexcept {
      return dequeued_;
    }

    size_t GateBase::OpenRequest::processed() const noexcept {
      return dequeued() + available_rounds();
    }

    const Tensor *GateBase::OpenRequest::id_and_count(OpKernelContext *ctx) {
        return id_and_count_.AccessTensor(ctx);
    }

    void GateBase::OpenRequest::set_id_and_count_batching(OpKernelContext *ctx, size_t batch_size) {
      DCHECK_GT(batch_size, 0);
      DCHECK_EQ(dequeued(), 0);
      auto idc_t = id_and_count_.AccessTensor(ctx);
      auto idc = idc_t->matrix<int32>();
      auto last_dim = idc_t->dim_size(0) - 1;

      for (int i = 0; i < idc.dimension(1); ++i) {
        auto result = div(idc(i, 1), batch_size);
        idc(i, 1) = result.rem > 0 ? result.quot + 1 : result.quot;
      }
    }

    GateBase::CreditRequest::CreditRequest(OpKernelContext *ctx, GateInterface::CallbackWithCredits &&callback,
                                           CancellationManager *cm, CancellationToken ct)
    : ctx_(ctx), ct_(ct), cm_(cm), callback_(move(callback)) {}

    bool GateBase::CreditRequest::matches_cancellation(CancellationManager *cm, CancellationToken ct) const noexcept {
      return cm == cm_ and ct == ct_;
    }

    GateInterface::CallbackWithCredits &GateBase::CreditRequest::get_callback() {
      return callback_;
    }

    void GateBase::CreditRequest::cancel(const std::string &gate_name) {
      DCHECK_NE(ctx_, nullptr);
      ctx_->SetStatus(Cancelled(gate_name, " credit request cancelled"));
    }

    void GateBase::CreditRequest::add_to_cleanup(vector <GateBase::CleanUp> &cleanup, int32 available_credits) const {
      DCHECK_GT(available_credits, 0);
      cleanup.emplace_back([callback(callback_), available_credits]() { callback(available_credits); },
                           ct_, cm_);
    }

    void GateBase::CreditRequest::add_error_to_cleanup(vector<GateBase::CleanUp> &cleanup) const {
      cleanup.emplace_back([callback(callback_)]() { callback(-1); }, ct_, cm_);
    }

    // Note: this is for the external facing one. Don't call this internally!
    // in case of double locking
    size_t GateBase::OpenRequestCount() const {
      mutex_lock l(mu_);
      return num_opened_requests();
    }

    size_t GateBase::NumOpenRounds() const {
      mutex_lock l(mu_);
      size_t open_rounds = 0;
      for (auto const &open_round_pair : request_map_) {
        auto const &open_round = open_round_pair.second;
        open_rounds += open_round.available_rounds();
      }
      return open_rounds;
    }

    string GateBase::LogGateState() const noexcept {
      stringstream s;

      s << name_ << ": enq_attempts: " << enqueue_attempts_.size() << ", deq_attempts: " << dequeue_attempts_.size()
              << ", credit_req " << (credit_request_.is_initialized() ? "" : "not ") << "set";

      if (limit_requests_from_upstream_gate_) {
        s << ", upstream " << opened_requests_ << "/" << max_open_requests_;
      }
      if (limit_requests_to_downstream_gate_) {
        s << ", downstream " << available_downstream_requests_;
      }
      auto open_req = num_opened_requests();

      if (open_req > 0) {
        s << " | ";

        size_t i = 0;
        for (auto const &open_round_pair : request_map_) {
          auto const &open_round = open_round_pair.second;
          s << open_round.id() << ":" << open_round.processed() << "/" << open_round.count() << "."
            << open_round.dequeued();
          if (i++ < open_req) {
            s << ", ";
          }
        }
      }
      return s.str();
    }

}  // namespace tensorflow
