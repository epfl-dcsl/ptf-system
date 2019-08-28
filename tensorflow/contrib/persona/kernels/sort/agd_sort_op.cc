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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_pair.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"

namespace tensorflow {

   namespace { 
      void resource_releaser(ResourceContainer<Data> *data) {
        core::ScopedUnref a(data);
        {
          ResourceReleaser<Data> rr(*data);
          data->get()->release();
        }
      }
   }

  using namespace std;
  using namespace errors;

  inline bool operator<(const Position& lhs, const Position& rhs) {
    if (lhs.ref_index() < rhs.ref_index()) {
      return true;
    } else if (lhs.ref_index() == rhs.ref_index()) {
      if (lhs.position() < rhs.position()) return true;
      else return false;
    } else
      return false;
  }

  class AGDSortOp : public OpKernel {
  public:
    AGDSortOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    ~AGDSortOp() {
      core::ScopedUnref unref_listpool(bufferpair_pool_);
    }

    Status GetOutputBufferPairs(OpKernelContext* ctx, int num, vector<ResourceContainer<BufferPair>*>& ctrs)
    {
      Tensor* bufs_out_t;
      TF_RETURN_IF_ERROR(ctx->allocate_output("partial_handle", TensorShape({num, 2}), &bufs_out_t));
      auto bufs_out = bufs_out_t->matrix<string>();

      ctrs.resize(num);
      for (size_t i = 0; i < num; i++) {

        TF_RETURN_IF_ERROR(bufferpair_pool_->GetResource(&ctrs[i]));
        ctrs[i]->get()->reset();
        bufs_out(i, 0) = ctrs[i]->container();
        bufs_out(i, 1) = ctrs[i]->name();
      }
      //TF_RETURN_IF_ERROR((*ctr)->allocate_output("partial_handle", ctx));
      return Status::OK();
    }

    Status LoadColumnDataResources(OpKernelContext *ctx, const Tensor *handles_t,
                                   vector<vector<AGDRecordReader>> &matrix, const Tensor *num_records_t,
                                   vector<unique_ptr<ResourceContainer<Data>, decltype(resource_releaser) * >> &releasers) {
      DCHECK(matrix.empty());
      auto rmgr = ctx->resource_manager();
      auto handles_tensor = handles_t->tensor<string, 3>();
      auto num_records = num_records_t->vec<int32>();
      ResourceContainer<Data> *input;

      auto num_groups = handles_tensor.dimension(0),
              num_columns = handles_tensor.dimension(1);
      for (int column = 0; column < num_columns; column++) {
        vector<AGDRecordReader> column_vec;
        column_vec.reserve(num_groups);
        for (int group = 0; group < num_groups; group++) {
          TF_RETURN_IF_ERROR(rmgr->Lookup(handles_tensor(group, column, 0), handles_tensor(group, column, 1), &input));
          column_vec.emplace_back(input, num_records(group));
          releasers.push_back(move(vector<unique_ptr<ResourceContainer<Data>, decltype(resource_releaser)&>>::value_type(input, resource_releaser)));
        }
        matrix.push_back(move(column_vec));
      }
      return Status::OK();
    }

    Status LoadSortKeyDataResources(OpKernelContext *ctx, const Tensor *handles_t,
                                    vector<AGDRecordReader> &vec, const Tensor *num_records_t,
                                    vector<unique_ptr<ResourceContainer<Data>, decltype(resource_releaser) * >> &releasers) {
      auto rmgr = ctx->resource_manager();
      auto handles_matrix = handles_t->matrix<string>();
      auto num = handles_t->shape().dim_size(0);
      auto num_records = num_records_t->vec<int32>();
      ResourceContainer<Data> *input;

      for (int i = 0; i < num; i++) {
        TF_RETURN_IF_ERROR(rmgr->Lookup(handles_matrix(i, 0), handles_matrix(i, 1), &input));
        vec.push_back(AGDRecordReader(input, num_records(i)));
        releasers.push_back(move(vector<unique_ptr<ResourceContainer<Data>, decltype(resource_releaser)&>>::value_type(input, resource_releaser)));
      }
      return Status::OK();
    }
    

    void Compute(OpKernelContext* ctx) override {
      if (!bufferpair_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_pair_pool", &bufferpair_pool_));
      }

      sort_index_.clear();

      const Tensor *results_in, *columns_in, *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->vec<int32>();
      OP_REQUIRES_OK(ctx, ctx->input("sort_key_handles", &results_in));
      OP_REQUIRES_OK(ctx, ctx->input("column_handles", &columns_in));
      int64 num_columns = columns_in->dim_size(1);
      int64 num_groups = columns_in->dim_size(0);
      int64 num_results = results_in->dim_size(0);
      auto num_records_length = num_records.size();

      // We have to validate the dynamic (unknown) sizes.
      OP_REQUIRES(ctx, num_groups == num_results, InvalidArgument("Got ", num_results, " results groups and ", num_groups, " other column groups. They should be equal!"));
      OP_REQUIRES(ctx, num_results == num_records_length, InvalidArgument("Got ", num_results, " results columns and ", num_records_length, " num_recs. They should be equal!"));

      vector<unique_ptr<ResourceContainer<Data>, decltype(resource_releaser)*>> releasers;

      int superchunk_records = 0;
      for (int i = 0; i < num_records_length; i++)
        superchunk_records += num_records(i);

      Tensor* records_out_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("superchunk_records", TensorShape({}), &records_out_t));
      records_out_t->scalar<int32>()() = superchunk_records;

      vector<AGDRecordReader> results_vec;
      OP_REQUIRES_OK(ctx, LoadSortKeyDataResources(ctx, results_in, results_vec, num_records_t, releasers));

      // phase 1: parse results sequentially, build up vector of (genome_location, index)
      Alignment agd_result;
      const char* data;
      size_t size;

      // only do this check to avoid allocation on reserve every time (which vector<> is allowed to do)
        /*
      auto resulting_num_records = num_results * num_records(0);
      if (sort_index_.capacity() < resulting_num_records)
          sort_index_.reserve(resulting_num_records);
          */

      for (int i = 0; i < num_results; i++) {
        auto& result_reader = results_vec[i];
        int j = 0;
        while(result_reader.GetNextRecord(&data, &size).ok()) { // only ResourceExhausted can be thrown. no need to check for other errors
          OP_REQUIRES(ctx, agd_result.ParseFromArray(data, size),
                      Internal("Could not parse results protobuf. Got string '", string(data, size), "'"));
          sort_index_.emplace_back(agd_result.position(), i, j++);
        }
      }

      // phase 2: sort the vector by genome_location
      //LOG(INFO) << "running std sort on " << sort_index_.size() << " SortEntry's";
      std::sort(sort_index_.begin(), sort_index_.end(), [](const SortEntry& a, const SortEntry& b) {
          return a.position < b.position;
          });

      // phase 3: using the sort vector, merge the chunks into superchunks in sorted
      // order

      // now we need all the chunk data
      vector<vector<AGDRecordReader>> columns_matrix;
      //columns_matrix.resize(num_columns);
      OP_REQUIRES_OK(ctx, LoadColumnDataResources(ctx, columns_in, columns_matrix, num_records_t, releasers));

      // get output buffer pairs (pair holds [index, data] to construct
      // AGD format temp output file in next dataflow stage)
      vector<ResourceContainer<BufferPair>*> bufpair_containers;
      OP_REQUIRES_OK(ctx, GetOutputBufferPairs(ctx, num_columns+1, bufpair_containers));

      vector<ColumnBuilder> builders;
      // +1 for results
      builders.resize(num_columns+1);
      for (int i = 0; i < num_columns+1; i++) {
        builders[i].SetBufferPair(bufpair_containers[i]->get());
      }

      // column ordering will be [ results, <everything else> ]
      for (size_t i = 0; i < sort_index_.size(); i++) {
        auto& entry = sort_index_[i];
        auto& result_reader = results_vec.at(entry.chunk);
        result_reader.GetRecordAt(entry.index, &data, &size);
        builders[0].AppendRecord(data, size);

        for (size_t j = 0; j < num_columns; j++) {
          auto &column = columns_matrix.at(j);
          auto &reader = column.at(entry.chunk);
          reader.GetRecordAt(entry.index, &data, &size);
          builders[j+1].AppendRecord(data, size);
        }
      }
    }

  private:
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;

    struct SortEntry {
      // should double check actual size of this, using a protobuf object here might
      // not be a good idea
      Position position;
      uint8_t chunk;
      int index;
        SortEntry(decltype(position) p, decltype(chunk) c, decltype(index) i)
                : position(p), chunk(c), index(i) {}
    };

    vector<SortEntry> sort_index_;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDSort").Device(DEVICE_CPU), AGDSortOp);
} //  namespace tensorflow {
