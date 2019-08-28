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
#include <vector>
#include <memory>
#include <chrono>
#include <queue>
#include <functional>
#include <algorithm>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column.h"
#include "tensorflow/contrib/persona/kernels/agd-format/results_index.h"

#define CUSTOM_HEAP

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;


  inline bool operator>(const Position& lhs, const Position& rhs) {
    if (lhs.ref_index() > rhs.ref_index()) {
      return true;
    } else if (lhs.ref_index() == rhs.ref_index()) {
      if (lhs.position() > rhs.position()) return true;
      else return false;
    } else
      return false;
  }
  namespace {
    const string op_name("AGDMerge");

    void ResultsIndexReleaser(ResourceContainer<ResultsIndex> *ri) {
      ResourceReleaser<ResultsIndex> rr(*ri);
      ri->get()->reset();
    }

    TensorShape scalar_shape{};

    class ColumnCursor {
    public:
      ColumnCursor(ResultsIndex &results, vector<reference_wrapper<Column>> &&all_columns) :
        results_index_(results), all_columns_(move(all_columns)) {}

      bool advance() {
        auto ret =  results_index_.advance();
        /*
        if (ret) {
          auto const &l = get_location();
          LOG(INFO) << "Advancing to " << l.first << ":" << l.second;
        }
         */
        return ret;
      }

      bool append_to_buffer_pairs(vector<BufferPair *> &bp_vec) {
          DCHECK_EQ(bp_vec.size(), all_columns_.size());
          auto num_columns = all_columns_.size();
        for (size_t bl_idx = 0; bl_idx < num_columns; ++bl_idx) {
          auto &r = all_columns_[bl_idx];
          auto bp = bp_vec[bl_idx];
          if (not copy_record(bp, r)) {
            return false;
          }
        }

        return true;
      }

      inline const FastResult& get_location() {
        return results_index_.current_position();
      }
        inline const FastResult* unsafe_get_location() {
          return results_index_.unsafe_current_position();
        }

    private:
      vector<reference_wrapper<Column>> all_columns_;
      ResultsIndex &results_index_;

      static inline
      bool
      copy_record(BufferPair *bp, Column &column) {
        const char *record_data = nullptr;
        size_t record_size = 0;

        if (not column.GetNextRecord(&record_data, &record_size)) {
          return false;
        }
        auto idx_sz = static_cast<RelativeIndex>(record_size);
        auto &index = bp->index();
        auto &data = bp->data();
        //auto begin = chrono::high_resolution_clock::now();
        auto success = index.AppendBuffer(reinterpret_cast<const char *>(&idx_sz), sizeof(idx_sz)) and
                       data.AppendBuffer(record_data, record_size);
        /*
        auto end = chrono::high_resolution_clock::now();
        auto diff = chrono::duration_cast<chrono::microseconds>(end - begin).count();

        if (diff > 100) {
          LOG(INFO) << "Got a long buffer append of " << diff << " us";
        }
         */
        return success;
      }
    };

    typedef pair<const FastResult*, ColumnCursor*> GenomeScore;
    // a > b because we want a min-heap.
    bool CompareScore(const GenomeScore &a, const GenomeScore &b) {
        auto const &af = a.first;
        auto const &bf = b.first;
        auto aff = af->first;
        auto bff = bf->first;
        return aff > bff or (aff == bff and af->second > bf->second);
      //return a.first > b.first;
    }

#ifdef CUSTOM_HEAP
    class MinPositionHeap {
    public:
        MinPositionHeap(size_t max_num_items) {
          scores_.reserve(max_num_items);
        }

        void AddItem(GenomeScore &&gs) {
          scores_.emplace_back(move(gs));
        }

        void Heapify() {
          DCHECK(not empty());
          make_heap(scores_.begin(), scores_.end(), CompareScore);
        }

        void RemoveMin() {
          pop_heap(scores_.begin(), scores_.end(), CompareScore);
          scores_.pop_back();
        }

        void UpdateMin(const FastResult *fr) {
          DCHECK(not empty());
          scores_[0].first = fr;
          reheapify();
        }

        ColumnCursor *Min() const noexcept {
          DCHECK(not empty());
          return scores_[0].second;
        }

        bool empty() const noexcept {
          return scores_.empty();
        }
    private:
        vector<GenomeScore> scores_;

        void reheapify() {
          auto max_idx = scores_.size();
          decltype(max_idx) current = 0, child, smallest;
          while (true) {
            DCHECK_LT(current, max_idx);
            smallest = current;
            child = (current * 2) +1;
            if (child < max_idx) {
              if (CompareScore(scores_[smallest], scores_[child])) {
                smallest = child;
              }
              child++;
              if (child < max_idx and CompareScore(scores_[smallest], scores_[child])) {
                smallest = child;
              }
            }

            if (smallest == current) {
              break;
            } else {
              swap(scores_[current], scores_[smallest]);
              current = smallest;
            }
          }
        }
    };
#endif
  }

  class AGDMergeOp : public OpKernel {
  public:
    AGDMergeOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
    }

    ~AGDMergeOp() override {
      core::ScopedUnref queue_unref(queue_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!queue_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      auto rsrc_mgr = ctx->resource_manager();

      const Tensor *chunk_group_handles_t, *results_index_t;
      OP_REQUIRES_OK(ctx, ctx->input("chunk_group_handles", &chunk_group_handles_t));
      OP_REQUIRES_OK(ctx, ctx->input("results_indexes", &results_index_t));
      auto chunk_group_shape = chunk_group_handles_t->shape();
      auto num_super_chunks = chunk_group_shape.dim_size(0);
      auto num_columns = chunk_group_shape.dim_size(1);
      auto chunk_group_handles = chunk_group_handles_t->tensor<string, 3>();

      auto results_index_s = results_index_t->matrix<string>();
      auto rdim0 = results_index_s.dimension(0);
      auto cgdim0 = chunk_group_handles.dimension(0);
      OP_REQUIRES(ctx, rdim0 == cgdim0, InvalidArgument(
              "Results index column has dimension ", rdim0, " while chunk group has dimension ", cgdim0, " in dim 0. Must be equal!"
      ));

      OpInputList other_components;
      OP_REQUIRES_OK(ctx, ctx->input_list("other_components", &other_components));

      vector<ColumnCursor> columns;
      vector<unique_ptr<ResourceContainer<Column>, decltype(ColumnResourceReleaser)*>> releasers;

      // Note: we don't keep the actual ColumnCursors in here. all the move and copy ops would get expensive!
#ifdef CUSTOM_HEAP
      MinPositionHeap min_heap(num_super_chunks);
#else
      priority_queue<GenomeScore, vector<GenomeScore>, decltype(CompareScore)*> score_heap(CompareScore);
#endif
      releasers.reserve(num_super_chunks * num_columns);
      columns.reserve(num_super_chunks);
      bool success = false;
      ResourceContainer<Column> *data;

      ResourceContainer<ResultsIndex> *index_container;
      vector<unique_ptr<remove_pointer<decltype(index_container)>::type, decltype(ResultsIndexReleaser)*>> results_index_releasers;

      size_t total_records = 0;
      vector<size_t> column_sizes;
      for (decltype(num_super_chunks) super_chunk = 0; super_chunk < num_super_chunks; ++super_chunk) {
        OP_REQUIRES_OK(ctx, rsrc_mgr->Lookup(results_index_s(super_chunk, 0),
                                             results_index_s(super_chunk, 1), &index_container));
        results_index_releasers.push_back(move(decltype(results_index_releasers)::value_type(index_container, ResultsIndexReleaser)));
        auto &index = *index_container->get();
        decltype(num_columns) column = 0;

        auto index_num_records = index.size();
        DCHECK_GT(index_num_records, 0);
        total_records += index_num_records;

        // Then we look up the rest of the columns
        vector<reference_wrapper<Column>> all_columns;
        all_columns.reserve(num_columns);
        for (column = 0; column < num_columns; ++column) {
          OP_REQUIRES_OK(ctx, rsrc_mgr->Lookup(chunk_group_handles(super_chunk, column, 0),
                                               chunk_group_handles(super_chunk, column, 1), &data));
          auto &other_column = *data->get();
          auto other_num_records = other_column.NumRecords();
          OP_REQUIRES(ctx, other_num_records == index_num_records, Internal("Got an unmatched number of records comparing results column (",
                                                                                      index_num_records, ") and another column (", other_num_records, ")"));
          all_columns.push_back(other_column);
          releasers.push_back(move(decltype(releasers)::value_type(data, ColumnResourceReleaser)));
        }

        if (column_sizes.empty()) {
          column_sizes.reserve(num_columns);
          for (decltype(num_columns) column = 0; column < num_columns; column++) {
              auto &column_handle = all_columns[column].get();
              const char *data;
              size_t data_size;
              OP_REQUIRES(ctx, column_handle.PeekNextRecord(&data, &data_size), Internal("Couldn't peek size for column ", column));
              column_sizes.push_back((data_size*1.2) * chunk_size_);
          }
        }

        ColumnCursor a(index, move(all_columns));
        columns.push_back(move(a));
      }

      // rounding up per the specification
      const int32 num_chunks = (total_records / chunk_size_) + (total_records % chunk_size_ == 0 ? 0 : 1);

      // Now that everything is initialized, add the scores to the heap
      for (auto &cc : columns) {
#ifdef CUSTOM_HEAP
        min_heap.AddItem(GenomeScore(cc.unsafe_get_location(), &cc));
#else
        score_heap.emplace(cc.unsafe_get_location(), &cc);
#endif
      }

      int32 current_chunk_size = 0;
      int64 ordinal = 0;
      ColumnCursor *cc;
      vector<ResourceContainer<BufferPair>*> bp_ctrs;
      vector<BufferPair*> bufferpairs;
      OP_REQUIRES_OK(ctx, AcquireRow(bp_ctrs, bufferpairs, num_columns));

#ifdef CUSTOM_HEAP
        while (not min_heap.empty()) {
#else
      while (not score_heap.empty()) {
#endif
#ifdef CUSTOM_HEAP
          cc = min_heap.Min();
#else
          auto &top = score_heap.top();
          cc = top.second;
#endif

        OP_REQUIRES(ctx, cc->append_to_buffer_pairs(bufferpairs), Internal("Unable to append into non-empty buffer pairs"));
#ifndef CUSTOM_HEAP
        score_heap.pop();
#endif

        if (cc->advance()) { // true if it has another record
#ifdef CUSTOM_HEAP
            min_heap.UpdateMin(cc->unsafe_get_location());
            } else {
          min_heap.RemoveMin();
#else
            score_heap.emplace(cc->unsafe_get_location(), cc);
#endif
        }

        // pre-increment because we just added 1 to the chunk size
        // we're guaranteed that chunk size is at least 1
        if (++current_chunk_size == chunk_size_) {
          OP_REQUIRES_OK(ctx, EnqueueOutput(ctx, bp_ctrs, current_chunk_size, ordinal, num_chunks, other_components));
          OP_REQUIRES_OK(ctx, AcquireRow(bp_ctrs, bufferpairs, num_columns));
          ordinal += current_chunk_size; // only bother incrementing this on actually writing it out
          current_chunk_size = 0;
        }
      }

      if (current_chunk_size > 0) {
        // no need to bump ordinal here
        OP_REQUIRES_OK(ctx, EnqueueOutput(ctx, bp_ctrs, current_chunk_size, ordinal, num_chunks, other_components));
      }

      //end = chrono::high_resolution_clock::now();
      //auto merge_duration = chrono::duration_cast<chrono::microseconds>(end - real_begin);
      //LOG(INFO) << name() << " duration: " << merge_duration.count() / 1000.0 << "ms";
    }

  private:
    QueueInterface *queue_ = nullptr;
    ReferencePool<BufferPair> *bufpair_pool_ = nullptr;
    int chunk_size_;

    Status Init(OpKernelContext *ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 1), &queue_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pair_pool", &bufpair_pool_));
      return Status::OK();
    }

     Status AcquireRow(vector<ResourceContainer<BufferPair>*> &row_containers, vector<BufferPair*> &raw, const size_t num_columns) {
        raw.clear();
        row_containers.clear();
        for (remove_const<decltype(num_columns)>::type i = 0; i < num_columns; ++i) {
            remove_reference<decltype(row_containers)>::type::value_type rc;
            TF_RETURN_IF_ERROR(bufpair_pool_->GetResource(&rc));
            raw.emplace_back(rc->get());
            row_containers.emplace_back(move(rc));
        }
        return Status::OK();
    }

    Status EnqueueOutput(OpKernelContext *ctx, vector<ResourceContainer<BufferPair> *> &bp_ctrs, int32 num_records,
                         int64 ordinal, int32 num_chunk_files, const OpInputList &other_components) {
      QueueInterface::Tuple tuple;
      Tensor num_recs_out, ordinal_out, num_chunks_out;

      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT32, scalar_shape, &num_recs_out));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT64, scalar_shape, &ordinal_out));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT32, scalar_shape, &num_chunks_out));
      num_recs_out.scalar<int32>()() = num_records;
      ordinal_out.scalar<int64>()() = ordinal;
      num_chunks_out.scalar<int32>()() = num_chunk_files;

      Tensor buffer_container_t;
      auto num_columns = bp_ctrs.size();
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, {num_columns, 2}, &buffer_container_t));
      auto buffer_container = buffer_container_t.matrix<string>();
      for (size_t i = 0; i < num_columns; ++i) {
          auto &bpc = bp_ctrs[i];
          auto bpp = bpc->get();
        buffer_container(i, 0) = bp_ctrs[i]->container();
        buffer_container(i, 1) = bp_ctrs[i]->name();
      }

      tuple.push_back(move(buffer_container_t));
      tuple.push_back(move(num_recs_out));
      tuple.push_back(move(ordinal_out));
      tuple.push_back(move(num_chunks_out));

      for (auto &other_component : other_components) {
        tuple.push_back(other_component);
      }

      TF_RETURN_IF_ERROR(queue_->ValidateTuple(tuple));

      // This is the synchronous version
      Notification n;
      queue_->TryEnqueue(tuple, ctx, [&n]() { n.Notify(); });
      n.WaitForNotification();

      return Status::OK();
    }

  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDMergeOp);
} // namespace tensorflow {
