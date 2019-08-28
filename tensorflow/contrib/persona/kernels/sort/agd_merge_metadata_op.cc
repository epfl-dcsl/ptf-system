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
#include <utility>
#include <queue>
#include <chrono>
#include <functional>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column.h"

#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"


namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;

  namespace {
    const string op_name("AGDMergeMetadata");
    TensorShape scalar_shape{};

    class ColumnCursor {
    public:
      ColumnCursor(Column &metadata, vector<reference_wrapper<Column>> &&other_columns) :
        metadata_(metadata), all_columns_(move(other_columns)) {}

      bool set_current_string() {
        const char* data;
        size_t data_sz;
        if (not metadata_.PeekNextRecord(&data, &data_sz)) {
            return false;
        }
        current_meta_ = data;
        current_size_ = data_sz;
        return true;
      }

      bool append_to_buffer_pairs(vector<BufferPair *> &bp_vec) {
        // first, dump the alignment result in the first column
        auto bp_metadata = bp_vec[0];
        if (not copy_record(bp_metadata, metadata_)) {
            return false;
        }

        size_t bl_idx = 1;
        for (auto &r : all_columns_) {
          auto bp = bp_vec[bl_idx++];
          if (not copy_record(bp, r)) {
              return false;
          }
        }

        return true;
      }

      inline const char* get_string(size_t &size) {
        size = current_size_;
        return current_meta_;
      }

    private:
      vector<reference_wrapper<Column>> all_columns_;
      Column &metadata_;
      const char * current_meta_ = nullptr;
      size_t current_size_;
      //int64_t current_location_ = -2048;

      static inline
      bool
      copy_record(BufferPair *bp, Column &r) {
        const char *record_data;
        size_t record_size;
        auto &index = bp->index();
        auto &data = bp->data();

        if (not r.GetNextRecord(&record_data, &record_size)) {
            return false;
        }
        RelativeIndex idx_sz = static_cast<RelativeIndex>(record_size);
        return index.AppendBuffer(reinterpret_cast<const char*>(&idx_sz), sizeof(idx_sz)) and data.AppendBuffer(record_data, record_size);
      }
    };

    typedef tuple<const char*, size_t, ColumnCursor*> MetadataScore;
    struct ScoreComparator {
      bool operator()(const MetadataScore &a, const MetadataScore &b) {
        return strncmp(get<0>(a), get<0>(b), min(get<1>(a), get<1>(b))) > 0;
      }
    };

  }

  class AGDMergeMetadataOp : public OpKernel {
  public:
    AGDMergeMetadataOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
    }

    ~AGDMergeMetadataOp() {
      core::ScopedUnref queue_unref(queue_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!queue_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

    auto begin_time = chrono::high_resolution_clock::now();
      const Tensor *chunk_group_handles_t;
      OP_REQUIRES_OK(ctx, ctx->input("chunk_group_handles", &chunk_group_handles_t));
      auto chunk_group_shape = chunk_group_handles_t->shape();
      auto num_super_chunks = chunk_group_shape.dim_size(0);
      auto num_columns = chunk_group_shape.dim_size(1);
      auto chunk_group_handles = chunk_group_handles_t->tensor<string, 3>();

        OpInputList other_components;
        OP_REQUIRES_OK(ctx, ctx->input_list("other_components", &other_components));

      auto rsrc_mgr = ctx->resource_manager();

      vector<ColumnCursor> columns;
      vector<unique_ptr<ResourceContainer<Column>, decltype(ColumnResourceReleaser)*>> releasers;

      // Note: we don't keep the actual ColumnCursors in here. all the move and copy ops would get expensive!
      priority_queue<MetadataScore, vector<MetadataScore>, ScoreComparator> score_heap;

      releasers.reserve(num_super_chunks * num_columns);
      columns.reserve(num_super_chunks);
      ResourceContainer<Column> *data;
      bool success = false;
      const char *meta;
      size_t size;

      size_t total_records = 0;
      for (decltype(num_super_chunks) super_chunk = 0; super_chunk < num_super_chunks; ++super_chunk) {
          decltype(num_columns) column = 0;
        // First, we look up the metadata column
        OP_REQUIRES_OK(ctx, rsrc_mgr->Lookup(chunk_group_handles(super_chunk, column, 0),
                                             chunk_group_handles(super_chunk, column, 1), &data));
          releasers.push_back(move(decltype(releasers)::value_type(data, ColumnResourceReleaser)));
          auto &metadata_column = *data->get();

        auto metadata_num_records = metadata_column.NumRecords();
        total_records += metadata_num_records;

        // Then we look up the rest of the columns
        vector<reference_wrapper<Column>> other_columns;
        other_columns.reserve(num_columns-1);
        for (column = 1; column < num_columns; ++column) {
          OP_REQUIRES_OK(ctx, rsrc_mgr->Lookup(chunk_group_handles(super_chunk, column, 0),
                                               chunk_group_handles(super_chunk, column, 1), &data));
            releasers.push_back(move(decltype(releasers)::value_type(data, ColumnResourceReleaser)));
            auto &other_column = *data->get();
          OP_REQUIRES(ctx, other_column.NumRecords() == metadata_num_records, Internal("Got an unmatched number of records comparing metadata column (",
                                                                                       metadata_num_records, ") and another column (", other_column.NumRecords(), ")"));
          other_columns.push_back(other_column);
        }

        ColumnCursor a(metadata_column, move(other_columns));
        OP_REQUIRES(ctx, a.set_current_string(), Internal("Unable to set the metadata string in initialization"));
        columns.push_back(move(a));
      }

      const int32 num_chunks = (total_records / chunk_size_) + (total_records % chunk_size_ == 0 ? 0 : 1);

      // Now that everything is initialized, add the scores to the heap
      for (auto &cc : columns) {
        meta = cc.get_string(size);
        score_heap.push(MetadataScore(meta, size, &cc));
      }

      int32 current_chunk_size = 0;
      int64 ordinal = 0;
      ColumnCursor *cc;
      vector<ResourceContainer<BufferPair>*> bp_ctrs;
      vector<BufferPair*> bufferpairs;
      bp_ctrs.resize(num_columns);
      for (auto& bp : bp_ctrs) {
        OP_REQUIRES_OK(ctx, bufpair_pool_->GetResource(&bp));
        bp->get()->reset();
        bufferpairs.push_back(bp->get());
      }

      Status s;
      while (not score_heap.empty()) {
        auto &top = score_heap.top();
        cc = get<2>(top);

        OP_REQUIRES(ctx, cc->append_to_buffer_pairs(bufferpairs), Internal("Unable to append to buffer pair"));

        //score_heap.pop();

        if (not cc->set_current_string()) {
          // get_location will have the location advanced by the append_to_buffer_list call above
        score_heap.pop();
        //  meta = cc->get_string(size);
        //  score_heap.push(MetadataScore(meta, size, cc));
        }

        // pre-increment because we just added 1 to the chunk size
        // we're guaranteed that chunk size is at least 1
        if (++current_chunk_size == chunk_size_) {
          OP_REQUIRES_OK(ctx, EnqueueOutput(ctx, bp_ctrs, current_chunk_size, ordinal, num_chunks, other_components));
          bufferpairs.clear();
          for (auto& bp : bp_ctrs) {
            OP_REQUIRES_OK(ctx, bufpair_pool_->GetResource(&bp));
            bp->get()->reset();
            bufferpairs.push_back(bp->get());
          }
            ordinal += current_chunk_size; // only bother incrementing this on actually writing it out
          current_chunk_size = 0;
        }
      }

      if (current_chunk_size > 0) {
          // no need to bump ordinal here
        OP_REQUIRES_OK(ctx, EnqueueOutput(ctx, bp_ctrs, current_chunk_size, ordinal, num_chunks, other_components));
      }
        auto end = chrono::high_resolution_clock::now();
        auto diff = end - begin_time;
        auto merge_duration = chrono::duration_cast<chrono::microseconds>(diff);
        LOG(INFO) << name() << " duration: " << merge_duration.count() / 1000.0 << "ms";
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

      Status EnqueueOutput(OpKernelContext *ctx, vector<ResourceContainer<BufferPair> *> &bp_ctrs, int32 num_records, int64 ordinal,
                           int32 num_chunk_files, const OpInputList &other_components) {
        QueueInterface::Tuple tuple;
        Tensor num_recs_out, first_ordinal_out, num_chunks_out;

        TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT32, scalar_shape, &num_recs_out));
        TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT64, scalar_shape, &first_ordinal_out));
          TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT32, scalar_shape, &num_chunks_out));
        num_recs_out.scalar<int32>()() = num_records;
        first_ordinal_out.scalar<int64>()() = ordinal;
          num_chunks_out.scalar<int32>()() = num_chunk_files;

        Tensor buffer_container_t;
        auto num_columns = bp_ctrs.size();
        TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, {num_columns, 2}, &buffer_container_t));
        auto buffer_container = buffer_container_t.matrix<string>();
        for (size_t i = 0; i < num_columns; ++i) {
          buffer_container(i, 0) = bp_ctrs[i]->container();
          buffer_container(i, 1) = bp_ctrs[i]->name();
        }

        tuple.push_back(move(buffer_container_t));
        tuple.push_back(move(num_recs_out));
        tuple.push_back(move(first_ordinal_out));
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

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDMergeMetadataOp);
} // namespace tensorflow {
