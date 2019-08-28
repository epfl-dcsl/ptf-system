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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

  using namespace errors;
  using namespace std;
  using namespace shape_inference;

  // let the consumer write their own doc
#define REGISTER_REFERENCE_POOL(_NAME) \
  REGISTER_OP(_NAME) \
  .Attr("size: int") \
  .Attr("bound: bool = true") \
  .Attr("container: string = ''") \
  .Attr("shared_name: string = ''") \
  .SetShapeFn([](InferenceContext* c) { \
      c->set_output(0, c->Vector(2)); \
      return Status::OK(); \
      }) \
  .Output("pool_handle: Ref(string)") \
  .SetIsStateful()

  Status check_vector(InferenceContext *c, size_t input_idx, size_t dim_size) {
    ShapeHandle input_data;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(input_idx), 1, &input_data));
    auto dim_handle = c->Dim(input_data, 0);
    auto dim_value = c->Value(dim_handle);
    if (dim_value != dim_size) {
      return Internal("Op expected tensor of size ", dim_size, ", but got ", dim_value);
    }
    return Status::OK();
  }

  Status check_scalar(InferenceContext *c, size_t input_idx) {
    ShapeHandle input_data;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(input_idx), 0, &input_data));
    return Status::OK();
  }

  REGISTER_OP("AGDAssembler")
    .Input("agd_read_pool: Ref(string)")
    .Input("base_handle: string")
    .Input("qual_handle: string")
    .Input("meta_handle: string")
    .Input("num_records: int32")
    .Output("agd_read_handle: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 4; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        TF_RETURN_IF_ERROR(check_scalar(c, 4));
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
  .Doc(R"doc(
Assembles all 3 fields (bases, qualities, and metadata) into a generic reader object
which is passed downstream for conversion / alignment.

Currently this op requires all 3 fields to be available.
If we need to only process a subset in the future, we must make a separate op.
)doc");

    REGISTER_OP("NoMetaAGDAssembler")
    .Input("agd_read_pool: Ref(string)")
    .Input("base_handle: string")
    .Input("qual_handle: string")
    .Input("num_records: int32")
    .Output("agd_read_handle: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 3; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        TF_RETURN_IF_ERROR(check_scalar(c, 3));
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
  .Doc(R"doc(
Assembles all 3 fields (bases, qualities, and metadata) into a generic reader object
which is passed downstream for conversion / alignment.

Currently this op requires all 3 fields to be available.
If we need to only process a subset in the future, we must make a separate op.
)doc");

    REGISTER_REFERENCE_POOL("AGDReadPool")
    .Doc(R"doc(
A pool specifically for agd read resources.

Intended to be used for AGDAssembler
)doc");

    REGISTER_OP("AGDCephMerge")
    .Attr("chunk_size: int >= 1")
    .Attr("intermediate_files: list(string)")
    .Attr("num_records: list(int)")
    .Attr("cluster_name: string")
    .Attr("user_name: string")
    .Attr("pool_name: string")
    .Attr("ceph_conf_path: string")
    .Attr("file_buf_size: int = 10")
    .Input("buffer_list_pool: Ref(string)")
    .Output("chunk_out: string")
    .Output("num_recs: int32")
    .SetIsStateful()
    .Doc(R"doc(
Merges multiple input chunks into chunks based on `chunk_size`
Only supports a single-stage of merging, i.e. this will not write out to an arbitrarily-large single chunk.

Each buffer list dequeued will have the same number of elements as the NUM_COLUMNS dimension for chunk_group_handles

chunk_size: the size, in number of records, of the output chunks
num_records: vector of number of records
file_buf_size: the buffer size used for each individual file, default 10MB.
)doc");

    REGISTER_OP("AGDCephWriteColumns")
    .Attr("cluster_name: string")
    .Attr("user_name: string")
    .Attr("ceph_conf_path: string")
    .Attr("compress: bool")
    .Attr("record_type: list({'raw','structured'})")
    .Input("output_queue_handle: resource")
    .Input("pool_name: string")
    .Input("record_id: string")
    .Input("column_handle: string")
    .Input("file_path: string")
    // TODO these can be collapsed into a vec(3) if that would help performance
    .Input("first_ordinal: int64")
    .Input("num_records: int32")
    .SetIsStateful() // TODO not sure if we need this
    .Doc(R"doc(
Writes out columns from a specified BufferList. The list contains
[data, index] BufferPairs. This Op constructs the header, unifies the buffers,
and writes to disk. Normally, this corresponds to a set of bases, qual, meta,
results columns.

This writes out to a Ceph object store only, defined by `cluster_name, user_name,
pool_name, and ceph_conf_path`.

Assumes that the record_id for a given set does not change for the runtime of the graph
and is thus passed as an Attr instead of an input (for efficiency);


)doc");

    REGISTER_OP("AGDGeneCoverage")
    .Attr("ref_sequences: list(string)")
    .Attr("ref_seq_sizes: list(int)")
    .Attr("scale: int")
    .Attr("max: int")
    .Attr("bg: bool")
    .Attr("d: bool")
    .Attr("dz: bool")
    .Attr("strand: string")
    .Attr("bga: bool")
    .Input("results_handle: string")
    .Input("num_records: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status::OK();
        })
    .SetIsStateful()
    .Doc(R"doc(
    Gives coverage values of each of the base-pair in reference genome.
    )doc");

    REGISTER_OP("ImportSGA")
    .Attr("ref_sequences: list(string)")
    .Attr("ref_seq_sizes: list(int)")
   // .Attr("scale: int")
   // .Attr("max: int")
   // .Attr("bg: bool")
   // .Attr("d: bool")
   // .Attr("dz: bool")
    .Attr("path: string")
    .Attr("feature: string")
    //.Attr("bga: bool")
    .Input("results_handle: string")
    .Input("num_records: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status::OK();
        })
    .SetIsStateful()
    .Doc(R"doc(
    Converts aligned AGD to SGA 
    )doc");

 REGISTER_OP("Compression")
    .Attr("ref_sequences: list(string)")
    .Attr("ref_seq_sizes: list(int)")
   // .Attr("scale: int")
   // .Attr("max: int")
   // .Attr("bg: bool")
   // .Attr("d: bool")
   // .Attr("dz: bool")
    .Attr("feature: string")
    //.Attr("bga: bool")
    .Input("results_handle: string")
    .Input("bases_handle: string")
    .Input("num_records: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status::OK();
        })
    .SetIsStateful()
    .Doc(R"doc(
    Compresses the bases 
    )doc");

    REGISTER_OP("AGDConverter")
    .Input("buffer_pair_pool: Ref(string)")
    .Input("input_data: string")
    .Output("bases_out: string")
    .Output("qual_out: string")
    .Output("meta_out: string")
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        c->set_output(0, c->Vector(2));
        c->set_output(1, c->Vector(2));
        c->set_output(2, c->Vector(2));

        return Status::OK();
        })
  .Doc(R"doc(
Converts an input file into three files of bases, qualities, and metadata
)doc");

    REGISTER_OP("AGDInterleavedConverter")
    .Input("buffer_pair_pool: Ref(string)")
    .Input("input_data_0: string")
    .Input("input_data_1: string")
    .Output("bases_out: string")
    .Output("qual_out: string")
    .Output("meta_out: string")
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 3; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        c->set_output(0, c->Vector(2));
        c->set_output(1, c->Vector(2));
        c->set_output(2, c->Vector(2));

        return Status::OK();
        })
  .Doc(R"doc(
Converts two input files into three files of interleaved bases, qualities, and metadata
)doc");

    REGISTER_OP("AGDMarkDuplicates")
    .Input("buffer_pair_pool: Ref(string)")
    .Input("results_handle: string")
    .Input("num_records: int32")
    .Output("marked_results: string")
    .SetShapeFn([](InferenceContext *c) {
        ShapeHandle input_data;
        for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));
        auto dim_handle = c->Dim(input_data, 0);
        auto dim_value = c->Value(dim_handle);
        if (dim_value != 2) {
        return Internal("AGDConverter input ", i, " must be a vector(2)");
        }
        }
        c->set_output(0, input_data);

        return Status::OK();
        })
  .SetIsStateful()
    .Doc(R"doc(
Mark duplicate reads/pairs that map to the same location.

This Op depends on data being sorted by metadata (QNAME),
i.e. A paired read is immediately followed by its mate.

Normally this step would be run on the aligner output before
sorting by genome location.

The implementation follows the approach used by SamBlaster
github.com/GregoryFaust/samblaster
wherein read pair signatures are looked up in a hash table
to determine if there are reads/pairs mapped to the exact
same location. Our implementation uses google::dense_hash_table,
trading memory for faster execution.
  )doc");

    REGISTER_OP("AGDFlagstat")
    .Input("results_handle: string")
    .Input("num_records: int32")
    .Output("result: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status::OK();
        })
  .Doc(R"doc(
Flagstat module that gathers and displays stats on a dataset
  )doc");

    Status merge_op_shape_fn(InferenceContext *c) {
        auto chunk_input = c->input(2);
        if (c->Rank(chunk_input) != 3) {
            return InvalidArgument("chunk_group_handles must be a 3-tensor. Got rank ", c->Rank(chunk_input));
        }

        auto input2 = c->Dim(chunk_input, 2);
        if (!(c->ValueKnown(input2) and c->Value(input2) == 2)) {
            return InvalidArgument("chunk_group_handles must have innermost dimension of size 2");
        }

        auto input0 = c->Dim(chunk_input, 1);
        if (!c->ValueKnown(input0)) {
            return InvalidArgument("number of columns of chunk_group_handles must be known (dimension 0)");
        }

        return Status::OK();
    }


    REGISTER_OP("AGDMergeMetadata")
    .Attr("chunk_size: int >= 1")
    .Input("buffer_pair_pool: Ref(string)")
    .Input("output_buffer_queue_handle: resource")
    .Input("chunk_group_handles: string")
    .Input("other_components: Tcomponents")
    .Attr("Tcomponents: list(type) >= 0")
    .SetIsStateful()
    .SetShapeFn(merge_op_shape_fn)
    .Doc(R"doc(
Merges multiple input chunks into chunks based on `chunk_size`, using the metadata field
as sort key.

The following items are enqueued into the output: a chunk of {meta, other, columns, ..., based on the input}, num_records, first_ordinal, num_chunks, [anything else you added in other_components]

Only supports a single-stage of merging, i.e. this will not write out to an arbitrarily-large single chunk.

Each buffer list dequeued will have the same number of elements as the NUM_COLUMNS dimension for chunk_group_handles

chunk_size: the size, in number of records, of the output chunks
)doc");

    REGISTER_OP("AGDMerge")
            .Attr("chunk_size: int >= 1")
            .Input("buffer_pair_pool: Ref(string)")
            .Input("output_buffer_queue_handle: resource")
            .Input("chunk_group_handles: string")
            .Input("results_indexes: string")
            .Input("other_components: Tcomponents")
            .Attr("Tcomponents: list(type) >= 0")
            .SetIsStateful()
            .SetShapeFn([](InferenceContext *c) {
                TF_RETURN_IF_ERROR(merge_op_shape_fn(c));
                ShapeHandle sh;
                TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &sh));
                auto input1 = c->Dim(sh, 1);
                if (not c->ValueKnown(input1)) {
                    return InvalidArgument("agd merge results index must have known value in dim 1");
                }
                auto i1_val = c->Value(input1);
                if (i1_val != 2) {
                    return InvalidArgument("agd merge results index must have value 2 for dim 1");
                }
                return Status::OK();
            })
    .Doc(R"doc(
Merges multiple input chunks into chunks based on `chunk_size`
Only supports a single-stage of merging, i.e. this will not write out to an arbitrarily-large single chunk.

The following items are enqueued into the output: a chunk of {results, other, columns, ..., based on the input}, num_records, first_ordinal, num_chunks, [anything else you added in other_components]

Each buffer list dequeued will have the same number of elements as the NUM_COLUMNS dimension for chunk_group_handles

chunk_size: the size, in number of records, of the output chunks
*_handles: matrix of processed handles
)doc");

    REGISTER_OP("AGDOutput")
    .Attr("unpack: bool = true")
    .Attr("columns: list(string)")
    .Input("chunk_names: string")
    .Input("chunk_size: int32")
    .Input("start: int32")
    .Input("finish: int32")
    .SetIsStateful()
    .Doc(R"doc(
Takes a vector of string keys for AGD chunks (full paths)

Prints records to stdout from record indices `start` to `finish`.
  )doc");

    REGISTER_OP("AGDBaseCompression")
    .Attr("unpack: bool = true")
    .Attr("columns: list(string)")
    .Input("chunk_names: string")
    .Input("chunk_size: int32")
    .Input("start: int32")
    .Input("finish: int32")
    .SetIsStateful()
    .Doc(R"doc(
     compresses the base string using the cigar
    )doc");

    REGISTER_OP("AGDBaseDecompression")
    .Attr("ref_sequences: list(string)")
    .Attr("ref_index: list(int)")
    .Attr("unpack: bool = true")
    .Attr("columns: list(string)")
    .Input("chunk_names: string")
    .Input("chunk_size: int32")
    .Input("start: int32")
    .Input("finish: int32")
    .SetIsStateful()
    .Doc(R"doc(
     decompress the compressed base string into its original string of basepairs.
     )doc");

    REGISTER_OP("AGDReader")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("verify: bool = false")
    .Attr("reserve: int = 8192")
    .Attr("unpack: bool = true")
    .Attr("repack: bool = false")
    .Input("buffer_pool: Ref(string)")
    .Input("file_handle: string")
    .Output("processed_buffers: string")
    .Output("num_records: int32")
    .Output("first_ordinal: int64")
    .Output("record_id: string")
    .SetShapeFn([](InferenceContext *c) {
        bool unpack, repack;
        TF_RETURN_IF_ERROR(c->GetAttr("unpack", &unpack));
        TF_RETURN_IF_ERROR(c->GetAttr("repack", &repack));
        int pack_commands = (unpack ? 1 : 0) + (repack ? 1 : 0);
        if (pack_commands == 2) {
            return InvalidArgument("Can't specify both unpack and repack in AGDReader");
        }

        ShapeHandle sh;
        TF_RETURN_IF_ERROR(check_vector(c, 1, 2));

        c->set_output(0, c->Vector(2));
        for (int i = 1; i < 4; i++) {
            c->set_output(i, c->Scalar());
        }

        return Status::OK();
        })
  .SetIsStateful()
    .Doc(R"doc(
Read in the agd format from an upstream source (file reader or network reader).

Outputs a handle to the buffer containing the processed data

Input buffer_pool is a handle to a tensorflow::BufferPoolOp result tensor,
and file_handle should come from a file_mmap_op

reserve: the number of bytes to call 'reserve' on the vector.
  )doc");

    Status verify_sort_shape(InferenceContext *c) {
        auto sort_key_input = c->input(1);
        if (c->Rank(sort_key_input) != 2) {
            return InvalidArgument("Expected sort_key_input to have rank 2, but has rank ", c->Rank(sort_key_input));
        }
        auto column_handles_input = c->input(2);
        if (c->Rank(column_handles_input) != 3) {
            return InvalidArgument("Expected column_handles_input to have rank 3, but has rank ", c->Rank(column_handles_input));
        }
        auto num_records_input = c->input(3);
        if (c->Rank(num_records_input) != 1) {
            return InvalidArgument("Expected num_records_input to have rank 1, but has rank ", c->Rank(num_records_input));
        }

        bool sort_key_known = c->ValueKnown(c->Dim(sort_key_input, 0)),
                column_handle_known = c->ValueKnown(c->Dim(column_handles_input, 0)),
                num_records_known = c->ValueKnown(c->Dim(num_records_input, 0));

        if (sort_key_known) {
            auto sort_key_dim = c->Value(c->Dim(sort_key_input, 0));
            if (!column_handle_known) {
                return InvalidArgument("Known sort key dimension of ", sort_key_dim, ", but known column_handles 0th dimension");
            } else if (!num_records_known) {
                return InvalidArgument("Known sort key dimension of ", sort_key_dim, ", but known num_records 0th dimension");
            }

            auto column_handles_dim = c->Value(c->Dim(column_handles_input, 0)),
                    num_records_dim = c->Value(c->Dim(num_records_input, 0));
            if (sort_key_dim != column_handles_dim) {
                return InvalidArgument("Known sort_key_dim 0th dim of ", sort_key_dim, ", but known 0th dim for column_handles of ", column_handles_dim);
            } else if (sort_key_dim != num_records_dim) {
                return InvalidArgument("Known sort_key_dim 0th dim of ", sort_key_dim, ", but known 0th dim for num_records of ", num_records_dim);
            }
        } else {
            if (column_handle_known) {
                return InvalidArgument("Unknown 0th dim for sort_key, but known dimension for column_handles");
            } else if (num_records_known) {
                return InvalidArgument("Unknown 0th dim for sort_key, but known dimension for num_records");
            }
        }
        auto column_count_dim = c->Dim(column_handles_input, 1);
        if (!c->ValueKnown(column_count_dim)) {
            return InvalidArgument("Number of columns must be specified");
        }
        auto column_count = c->Value(column_count_dim) + 1; // for the sort key results
        c->set_output(0, c->Matrix(column_count, 2));
        c->set_output(1, c->Scalar());
       return Status::OK();
    }

#define REGISTER_SORT_OP(NAME) \
    REGISTER_OP(NAME) \
    .Input("buffer_pair_pool: Ref(string)") \
    .Input("sort_key_handles: string") \
    .Input("column_handles: string") \
    .Input("num_records: int32") \
    .Output("partial_handle: string") \
    .Output("superchunk_records: int32") \
    .SetIsStateful() \
    .SetShapeFn(verify_sort_shape)

    REGISTER_SORT_OP("AGDSort")
    .Doc(R"doc(
Takes N results buffers, and associated chunk buffers, and sorts them into a merged a superchunk output buffer. This
is the main sort step in the AGD external merge sort.

Outputs handle to merged, sorted superchunks in `partial_handles`.
A BufferList that contains bases, qual, meta, results superchunk
BufferPairs ready for writing to disk.

Inputs -> (N, 2) string handles to buffers containing results, bases,
qualities and metadata. num_records is a vector of int32's with the
number of records per chunk.

Takes an arbitrary number of columns. sort_key must be the results type
  )doc");

    REGISTER_SORT_OP("AGDSortMetadata")
    .Doc(R"doc(
Takes N results buffers, and associated bases, qualities and metadata
chunks, and sorts them into a merged a superchunk output buffer. This
is the main sort step in the AGD external merge sort.

This version sorts by metadata (QNAME in SAM).

Outputs handle to merged, sorted superchunks in `partial_handles`.
A BufferList that contains bases, qual, meta, results superchunk
BufferPairs ready for writing to disk.

Inputs -> (N, 2) string handles to buffers containing results, bases,
qualities and metadata. num_records is a vector of int32's with the
number of records per chunk.

Takes an arbitrary number of columns. sort_key must be the metadata type
  )doc");

    REGISTER_OP("AGDVerifySort")
    .Input("path: string")
    .Input("chunk_names: string")
    .Input("chunk_sizes: int32")
    .SetIsStateful()
    .Doc(R"doc(
Verifies that the dataset referred to by `chunk_names` is sorted.

Chunk names must be in contiguous order.
  )doc");

    REGISTER_REFERENCE_POOL("BufferListPool")
    .Doc(R"doc(
Creates and initializes a pool containing a list of char buffers of size `buffer_size` bytes
  )doc");

    REGISTER_REFERENCE_POOL("BufferPairPool")
    .Doc(R"doc(
Creates and initializes a pool containing a pair of char buffers of size `buffer_size` bytes
  )doc");

    REGISTER_OP("BufferSink")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Input("data: string")
    .Doc(R"doc(
Consumes the buffer input and produces nothing
)doc");

    REGISTER_OP("BufferListSink")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Input("data: string")
    .Output("id: string")
    .Doc(R"doc(
Consumes the buffer input and produces nothing

Note that the output is meaningless. It's only purpose is so that
we can use it in other pipelines where writers are used
)doc");

    REGISTER_OP("CephReader")
    .Attr("cluster_name: string")
    .Attr("user_name: string")
    .Attr("ceph_conf_path: string")
    .Attr("read_size: int")
    .Attr("pool_name: string")
    .Attr("delete_after_read: bool = false")
    .Input("buffer_pool: Ref(string)")
    .Input("key: string")
    .Input("namespace: string")
    .Output("file_handle: string")
    .Output("time: int64")
    .Output("duration: int64")
    .Output("bytes: int32")
    .SetShapeFn([](InferenceContext *c) {
        ShapeHandle sh;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &sh));
        auto dim_handle = c->Dim(sh, 0);
        auto dim_val = c->Value(dim_handle);
        if (dim_val != 2) {
        return Internal("buffer_handle must have dimensions {2}. Got ", dim_val);
        }
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &sh));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &sh));
        c->set_output(0, c->input(0));
        for (int i = 1; i < 4; i++) {
            c->set_output(i, c->Scalar());
        }
        return Status::OK();
        })
  .Doc(R"doc(
Obtains file names from a queue, fetches those files from Ceph storage using Librados,
and writes them to a buffer from a pool of buffers.

buffer_pool: a handle to the buffer pool
key: key reference to the filename queue
file_handle: a Tensor(2) of strings to access the file resource in downstream nodes
delete_after_read: after the op reads it from ceph, it deletes the item
  )doc");

    REGISTER_OP("FastqChunker")
    .Attr("chunk_size: int >= 1")
    .Input("queue_handle: resource")
    .Input("fastq_file: string") 
    .Input("fastq_pool: Ref(string)")
    .SetShapeFn([](InferenceContext *c) {
        ShapeHandle fastq_file;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &fastq_file));
        auto dim_handle = c->Dim(fastq_file, 0);
        auto fastq_dim = c->Value(dim_handle);
        if (fastq_dim != 2) {
        return Internal("fastq_file requires 2-dimensional vector");
        }

        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &fastq_file));
        dim_handle = c->Dim(fastq_file, 0);
        fastq_dim = c->Value(dim_handle);
        if (fastq_dim != 2) {
        return Internal("fastq_pool requires 2-dimensional vector");
        }

        return Status::OK();
        })
  .Doc(R"doc(

)doc");

    REGISTER_OP("FastqInterleavedChunker")
    .Attr("chunk_size: int >= 1")
    .Input("queue_handle: resource")
    .Input("fastq_file_0: string")
    .Input("fastq_file_1: string") 
    .Input("fastq_pool: Ref(string)")
    .SetShapeFn([](InferenceContext *c) {
        ShapeHandle fastq_file;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &fastq_file));
        auto dim_handle = c->Dim(fastq_file, 0);
        auto fastq_dim = c->Value(dim_handle);
        if (fastq_dim != 2) {
        return Internal("fastq_file requires 2-dimensional vector");
        }
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &fastq_file));
        dim_handle = c->Dim(fastq_file, 0);
        fastq_dim = c->Value(dim_handle);
        if (fastq_dim != 2) {
        return Internal("fastq_file requires 2-dimensional vector");
        }

        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &fastq_file));
        dim_handle = c->Dim(fastq_file, 0);
        fastq_dim = c->Value(dim_handle);
        if (fastq_dim != 2) {
        return Internal("fastq_pool requires 2-dimensional vector");
        }

        return Status::OK();
    })
  .Doc(R"doc(

)doc");

    REGISTER_REFERENCE_POOL("FastqReadPool")
    .Doc(R"doc(
A pool to manage FastqReadResource objects
)doc");

    REGISTER_OP("FileMMap")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("synchronous: bool = false")
    .Attr("delete_after_use: bool = false")
    .Input("pool_handle: Ref(string)")
    .Input("filename: string")
    .Output("file_handle: string")
    .SetShapeFn([](InferenceContext* c) {
        TF_RETURN_IF_ERROR(check_vector(c, 0, 2));
        TF_RETURN_IF_ERROR(check_scalar(c, 1));
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
  .SetIsStateful()
    .Doc(R"doc(
Produces memory-mapped files, synchronously reads them, and produces a Tensor<2>
with the container and shared name for the file.

This is used in the case of a remote reader giving only the filenames to this reader
pool_handle: a handle to the filename queue
file_handle: a Tensor(2) of strings to access the shared mmaped file resource to downstream nodes
delete_after_use: if true, this will cause the file mmap to be deleted when it is done
filename: a Tensor() of string for the unique key for this file
  )doc");

    REGISTER_REFERENCE_POOL("MMapPool")
    .Doc(R"doc(
Creates pools of MemoryMappedFile objects
)doc");

    REGISTER_REFERENCE_POOL("BufferPool")
    .Doc(R"doc(
Creates and initializes a pool containing char buffers of size `buffer_size` bytes
  )doc");

    REGISTER_OP("AGDTester")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("sam_filename: string = ''")
    .Input("genome_handle: Ref(string)")
    .Input("agd_records: string")
    .Input("num_records: int32")
    .Output("agd_records_out: string")
    .Output("num_records_out: int32")
    .Doc(R"doc(
  Compares the agd format output with the SAM format output
)doc");

#define MAKE_OP(_name_)                         \
    REGISTER_OP(_name_)                           \
    .Output("handle: Ref(string)")                \
    .Attr("cmd_line: list(string)")                     \
    .Attr("container: string = ''")               \
    .Attr("shared_name: string = ''")             \
    .SetIsStateful()                              \
    .SetShapeFn([](InferenceContext *c) {         \
        c->set_output(0, c->Vector(2));           \
        return Status::OK();                      \
        })

    MAKE_OP("AlignerOptions")
    .Doc(R"doc(
An op that produces SNAP aligner options.
handle: The handle to the options.
cmd_line: The SNAP command line parsed to create the options.
container: If non-empty, this options is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this options will be shared under the given name
  across multiple sessions.
)doc");

    MAKE_OP("PairedAlignerOptions")
    .Doc(R"doc(
An op taht produces SNAP paired aligner options.
handle: The handle to the options.
cmd_line: The SNAP command line parsed to create the options.
container: If non-empty, this options is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this options will be shared under the given name
  across multiple sessions.
)doc");

    REGISTER_OP("GenomeIndex")
    .Output("handle: Ref(string)")
    .Attr("genome_location: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
  .Doc(R"doc(
    An op that creates or gives ref to a SNAP genome index.
    handle: The handle to the genomeindex resource.
    genome_location: The path to the genome index directory.
    container: If non-empty, this index is placed in the given container.
    Otherwise, a default container is used.
    shared_name: If non-empty, this queue will be shared under the given name
    across multiple sessions.
    )doc");

    REGISTER_OP("NullAligner")
    .Attr("subchunk_size: int >= 1")
    .Attr("wait_time_secs: float = 0.0")
    .Input("buffer_list_pool: Ref(string)")
    .Input("read: string")
    .Output("result_buf_handle: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        c->set_output(0, c->Matrix(1, 2));
        return Status::OK();
        })
  .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
wait_time specifies the minimum time that the alignment should take
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
outputs a tensor [num_reads] containing serialized reads and results
containing the alignment candidates.
)doc");

    REGISTER_OP("SnapAlignPaired")
    .Attr("subchunk_size: int >= 1")
    .Attr("max_secondary: int >= 0")
    .Attr("release_resources: bool = true")
    .Input("buffer_list_pool: Ref(string)")
    .Input("read: string")
    .Input("executor_handle: Ref(string)")
    .Output("result_buf_handle: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 3; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        int max_secondary = 0;
        TF_RETURN_IF_ERROR(c->GetAttr("max_secondary", &max_secondary));
        c->set_output(0, c->Matrix(1+max_secondary, 2));
        return Status::OK();
        })
  .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
outputs a tensor [num_reads] containing serialized reads and results
containing the alignment candidates.

Subchunk Size is the size in paired records. The actual chunk size will be 2x because of the pairing.
)doc");

    REGISTER_OP("SnapPairedExecutor")
    .Attr("num_threads: int >= 0")
    .Attr("work_queue_size: int >= 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Input("options_handle: Ref(string)")
    .Input("genome_handle: Ref(string)")
    .Output("executor_handle: Ref(string)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
  .Doc(R"doc(Provides a multithreaded execution context
to align paired reads using the SNAP algorithm.
  )doc");

    REGISTER_OP("SnapAlignSingle")
    .Attr("subchunk_size: int >= 1")
    .Attr("max_secondary: int >= 0")
    .Attr("release_resources: bool = true")
    .Input("buffer_list_pool: Ref(string)")
    .Input("read: string")
    .Input("executor_handle: Ref(string)")
    .Output("result_buf_handle: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 3; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        int max_secondary = 0;
        TF_RETURN_IF_ERROR(c->GetAttr("max_secondary", &max_secondary));

        c->set_output(0, c->Matrix(1+max_secondary, 2));
        return Status::OK();
        })
  .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
outputs a tensor [num_reads] containing serialized reads and results
containing the alignment candidates.
)doc");

    REGISTER_OP("SnapSingleExecutor")
    .Attr("num_threads: int >= 0")
    .Attr("work_queue_size: int >= 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Input("options_handle: Ref(string)")
    .Input("genome_handle: Ref(string)")
    .Output("executor_handle: Ref(string)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
  .Doc(R"doc(Provides a multithreaded execution context
to align single reads using the SNAP algorithm.
            )doc");

    REGISTER_OP("SnapIndexReferenceSequences")
    .Input("genome_handle: Ref(string)")
    .Output("ref_seqs: string")
    .Output("ref_lens: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        return Status::OK();
        })
    .Doc(R"doc(
    Given a SNAP genome index, produce a string containing the contigs
    (ref sequences) values comma separated.
    )doc");

    REGISTER_OP("BWASingleExecutor")
    .Attr("max_secondary: int >= 0")
    .Attr("num_threads: int >= 0")
    .Attr("work_queue_size: int >= 0")
    .Attr("max_read_size: int = 400")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Input("options_handle: Ref(string)")
    .Input("index_handle: Ref(string)")
    .Output("executor_handle: Ref(string)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
  .Doc(R"doc(Provides a multithreaded execution context
that aligns single reads using BWA. Pass to > 1 BWAAlignSingle nodes
for optimal performance.
            )doc");

    REGISTER_OP("BWAPairedExecutor")
    .Attr("max_secondary: int >= 0")
    .Attr("num_threads: int >= 0")
    .Attr("work_queue_size: int >= 0")
    .Attr("max_read_size: int = 400")
    .Attr("thread_ratio: float = 0.66")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Input("options_handle: Ref(string)")
    .Input("index_handle: Ref(string)")
    .Output("executor_handle: Ref(string)")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
  .Doc(R"doc(Provides a multithreaded execution context
that aligns paired reads using BWA. Pass to > 1 BWAAlignPaired nodes
for optimal performance.
            )doc");

    REGISTER_OP("BWAAlignSingle")
    .Attr("subchunk_size: int")
    .Attr("max_read_size: int = 400")
    .Attr("max_secondary: int >= 1")
    .Input("buffer_list_pool: Ref(string)")
    .Input("executor_handle: Ref(string)")
    .Input("read: string")
    .SetShapeFn([](InferenceContext* c) {
        int max_secondary = 0;
        TF_RETURN_IF_ERROR(c->GetAttr("max_secondary", &max_secondary));

        c->set_output(0, c->Matrix(1+max_secondary, 2));
        return Status::OK();
        })
  .Output("result_buf_handle: string")
    .SetIsStateful()
    .Doc(R"doc(
  Run single-ended alignment with BWA MEM. 
  max_secondary must be at least 1 for chimeric reads that BWA may output.
)doc");

    REGISTER_OP("BWAAlignPaired")
    .Attr("subchunk_size: int")
    .Attr("max_read_size: int = 400")
    .Attr("max_secondary: int >= 1")
    .Input("buffer_list_pool: Ref(string)")
    .Input("executor_handle: Ref(string)")
    .Input("read: string")
    .Output("result_buf_handle: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
        int max_secondary = 0;
        TF_RETURN_IF_ERROR(c->GetAttr("max_secondary", &max_secondary));

        c->set_output(0, c->Matrix(1+max_secondary, 2));
        return Status::OK();
        })
    .Doc(R"doc(
  Run single-ended alignment with BWA MEM. 
  max_secondary must be at least 1 for chimeric reads that BWA may output.
  Must use the BWA paired executor for `executor_handle`.
)doc");

    REGISTER_OP("BWAAssembler")
    .Input("bwa_read_pool: Ref(string)")
    .Input("base_handle: string")
    .Input("qual_handle: string")
    .Input("meta_handle: string")
    .Input("num_records: int32")
    .Output("bwa_read_handle: string")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
    .SetIsStateful()
    .Doc(R"doc(
Assembles all 3 fields (bases, qualities, and metadata) into a generic reader object
which is passed downstream for conversion / alignment.

Currently this op requires all 3 fields to be available.
If we need to only process a subset in the future, we must make a separate op.
)doc");

    REGISTER_OP("NoMetaBWAAssembler")
    .Input("bwa_read_pool: Ref(string)")
    .Input("base_handle: string")
    .Input("qual_handle: string")
    .Input("num_records: int32")
    .Output("bwa_read_handle: string")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
    .SetIsStateful()
    .Doc(R"doc(
Assembles all 3 fields (bases, qualities, and metadata) into a generic reader object
which is passed downstream for conversion / alignment.

Currently this op requires all 3 fields to be available.
If we need to only process a subset in the future, we must make a separate op.
)doc");

    REGISTER_REFERENCE_POOL("BWAReadPool")
    .Doc(R"doc(
A pool specifically for bwa read resources.

Intended to be used for BWAAssembler
)doc");

    REGISTER_OP("BWAIndex")
    .Attr("index_location: string")
    .Attr("ignore_alt: bool")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("handle: Ref(string)")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
    .SetIsStateful()
    .Doc(R"doc(
  An op that creates or gives ref to a bwa index.
  handle: The handle to the BWAIndex resource.
  genome_location: The path to the genome index directory.
  container: If non-empty, this index is placed in the given container.
  Otherwise, a default container is used.
  shared_name: If non-empty, this queue will be shared under the given name
  across multiple sessions.
  )doc");

    REGISTER_OP("BwaIndexReferenceSequences")
    .Input("index_handle: Ref(string)")
    .Output("ref_seqs: string")
    .Output("ref_lens: string")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        return Status::OK();
        })
    .SetIsStateful()
    .Doc(R"doc(
    Produces ref sequences and lengths in comma separated strings.
    )doc");

    REGISTER_OP("BWAOptions")
    .Output("handle: Ref(string)")
    .Attr("options: list(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Vector(2));
        return Status::OK();
        })
    .SetIsStateful()
    .Doc(R"doc(
  An op that creates or gives ref to a bwa index.
  handle: The handle to the BWAOptions resource.
  genome_location: The path to the genome index directory.
  container: If non-empty, this index is placed in the given container.
  Otherwise, a default container is used.
  shared_name: If non-empty, this queue will be shared under the given name
  across multiple sessions.
  )doc");

    REGISTER_OP("TwoBitConverter")
    .Input("num_records: int32")
    .Input("input: string")
    .Output("output: string")
    .SetShapeFn([](InferenceContext *c) {
        TF_RETURN_IF_ERROR(check_scalar(c, 0));
        TF_RETURN_IF_ERROR(check_vector(c, 1, 2));
        c->set_output(0, c->input(1));
        return Status::OK();
        })
  .Doc(R"doc(
Converts from an ASCII base buffer to a 2-bit output buffer, for BWA conversion.
This uses the same buffer, and can handle any Data type that exposes mutable access (e.g. Buffer)
)doc");

    REGISTER_OP("AgdImportBam")
    .Attr("path: string")
    .Attr("num_threads: int >= 1")
    .Attr("ref_seq_lens: list(int)")
    .Attr("chunk_size: int = 100000")
    .Attr("unaligned: bool = false")
    .Input("bufpair_pool: Ref(string)")
    .Output("chunk_out: string")
    .Output("num_records: int32")
    .Output("first_ordinal: int64")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        TF_RETURN_IF_ERROR(check_vector(c, 0, 2));
        bool unaligned;
        TF_RETURN_IF_ERROR(c->GetAttr("unaligned", &unaligned));
        int dim;
        if (unaligned) dim = 3;
        else dim = 4;
        c->set_output(0, c->Matrix(dim, 2));
        c->set_output(1, c->Scalar());
        c->set_output(2, c->Scalar());
        return Status::OK();
        })
  .Doc(R"doc(
Import AGD chunks from a BAM file. The BAM can be aligned or unaligned. 
If paired, sort order MUST be by ID (metadata).
This op (currently) will skip secondary or supplemental alignments.

path: the full path of the BAM file
num_threads: number of threads to give BAM reader
ref_seq_lens: vector of reference sequence lengths
chunk_size: the output dataset chunk size (default 100K)
unaligned: set to true if the bam file is unaligned (or you don't want to import results)
bufpair_pool: reference to buffer pair pool
chunk_out: a 3 or 4 x 2 matrix containing handles to chunks in buffer pairs
num_records: number of records in output. Usually `chunk_size` except for the last one
)doc");

    REGISTER_OP("AgdImportSra")
    .Attr("path: string")
    .Attr("num_threads: int >= 1")
    .Attr("chunk_size: int = 100000")
    .Attr("start: int = 0")
    .Attr("count: int >= 0")
    .Input("bufpair_pool: Ref(string)")
    .Output("chunk_out: string")
    .Output("num_records: int32")
    .Output("first_ordinal: int64")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        TF_RETURN_IF_ERROR(check_vector(c, 0, 2));
        int dim = 3;
        c->set_output(0, c->Matrix(dim, 2));
        c->set_output(1, c->Scalar());
        c->set_output(2, c->Scalar());
        return Status::OK();
        })
  .Doc(R"doc(
Import AGD chunks from a SRA file. 

path: the full path of the SRA file
num_threads: number of threads to give SRA reader
chunk_size: the output dataset chunk size (default 100K)
bufpair_pool: reference to buffer pair pool
chunk_out: a 3 x 2 matrix containing handles to chunks in buffer pairs
num_records: number of records in output. Usually `chunk_size` except for the last one
first_ordinal: ranges from 0 to the number of reads in the SRA file
)doc");

    REGISTER_OP("AgdOutputBam")
    .Attr("path: string")
    .Attr("pg_id: string")
    .Attr("ref_sequences: list(string)")
    .Attr("ref_seq_sizes: list(int)")
    .Attr("read_group: string")
    .Attr("sort_order: {'unknown', 'unsorted', 'queryname', 'coordinate'}")
    .Attr("num_threads: int >= 2")
    .Input("results_handle: string")
    .Input("bases_handle: string")
    .Input("qualities_handle: string")
    .Input("metadata_handle: string")
    .Input("num_records: int32")
    .Output("chunk: int32")
    .SetShapeFn([](InferenceContext* c) {
        for (size_t i = 1; i < 3; i++)
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        TF_RETURN_IF_ERROR(check_scalar(c, 4));
        c->set_output(0, c->Scalar());
        return Status::OK();
        })
    .SetIsStateful()
    .Doc(R"doc(
  On execution, append reads/results chunks to output BAM file.

  Not all tags for SAM/BAM are currently supported, but support
  is planned. Currently supported is only required tags.

  RG and aux data is currently not supported. 

  results_handle: matrix of all results columns
  path: path for output .bam file
  pg_id: program id @PG for .bam
  ref_sequences: Reference sequences, @RG tags.
  ref_seq_sizes: Sizes of the references sequences.
  read_group: read group tag @RG
  *handles: the records to append to the BAM file
  num_records: the number of records held in *handles
  num_threads: number of threads for compression >= 2 because one is
  used to coordinate writing to disk.
  )doc");

    // All the new prototypes of the write ops go here

#define AGD_COMMON_HEADER_ATTRIBUTES \
    .Attr("record_type: {'text', 'base_compact', 'structured'}") \
    .Input("path: string") \
    .Input("record_id: string") \
    .Input("first_ordinal: int64") \
    .Input("num_records: int32") \
    .Input("resource_handle: string") \
    .Output("output_path: string") \
    .SetIsStateful()

#define COMMON_AGD_DOC \
    "record_type: one of base, qual, meta, or results* " \
    "path: the string of the path / key to be written" \
    "record_id: the string to write into the header for this given record" \
    "first_ordinal: the first ordinal to write into the header" \
    "num_records: the number of records in this chunk" \
    "resource_handle: a Vec(2) to look up the resource containing the data to be written" \
    "path: the output path of the key / file that was written"

#define CEPH_WRITER_OP(WRITER_TYPE) \
    REGISTER_OP("AGDCeph" WRITER_TYPE "Writer") \
    .Attr("cluster_name: string") \
    .Attr("user_name: string") \
    .Attr("ceph_conf_path: string") \
    .Attr("pool_name: string") \
    .Input("namespace: string") \
    AGD_COMMON_HEADER_ATTRIBUTES \
    .Output("time: int64") \
    .Output("duration: int64") \
    .Output("bytes: int32") \
    .SetShapeFn([](InferenceContext *c) { \
        for (int i = 0; i < 5; i++) { \
        TF_RETURN_IF_ERROR(check_scalar(c, i)); \
        } \
        TF_RETURN_IF_ERROR(check_vector(c, 5, 2)); \
        for (int i = 0; i < 4; i++) {              \
          c->set_output(i, c->Scalar());           \
        }                                          \
        return Status::OK(); \
        }) \
    .Doc(R"doc( \
  Write a record of type " WRITER_TYPE " to Ceph \
   \
  cluster_name: Ceph cluster name \
  user_name: Ceph user name \
  ceph_conf_path: path to Ceph configuration file \
  pool_name: pool name to look up a given record)doc" \
    COMMON_AGD_DOC \
    )


#define FS_WRITER_OP(WRITER_TYPE) \
    REGISTER_OP("AGDFileSystem" WRITER_TYPE "Writer") \
    AGD_COMMON_HEADER_ATTRIBUTES \
    .SetShapeFn([](InferenceContext *c) { \
        for (int i = 0; i < 4; i++) { \
        TF_RETURN_IF_ERROR(check_scalar(c, i)); \
        } \
        TF_RETURN_IF_ERROR(check_vector(c, 4, 2)); \
        c->set_output(0, c->Scalar()); \
        return Status::OK(); \
        })

#define DUAL_WRITER_OP(WRITER_TYPE) \
    CEPH_WRITER_OP(WRITER_TYPE); \
    FS_WRITER_OP(WRITER_TYPE)

  DUAL_WRITER_OP("BufferPair");
  DUAL_WRITER_OP("BufferList");

  CEPH_WRITER_OP("Buffer")
    .Attr("compressed: bool");

  FS_WRITER_OP("Buffer")
    .Attr("compressed: bool");

    REGISTER_OP("StageBarrier")
    .Input("barrier_request_id: string")
    .Input("barrier_request_count: int32")
    .Input("input_queue_ref: resource")
    .Input("output_queue_ref: resource")
    .Output("request_id_out: string")
    .Output("request_count_out: int32")
    .SetShapeFn([](InferenceContext* c) {
        for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(check_scalar(c, 0));
        c->set_output(i, c->input(0));
        }
        return Status::OK();
        })
    .Doc(R"doc(
    Experimental.
    )doc");

    REGISTER_OP("Batcher")
    .Attr("batch_size: int >= 1")
    .Attr("T: type")
    .Attr("shape: shape")
    .Input("input_queue_ref: resource")
    .Output("batched_tensor: T")
    .Output("request_id: string")
    .SetShapeFn([](InferenceContext* c) {
        TensorShapeProto input_proto;
        TF_RETURN_IF_ERROR(c->GetAttr("shape", &input_proto));
        ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(input_proto, &input_shape));
        if (!c->FullyDefined(input_shape)) {
        return Internal("attr shape must be fully defined");
        }
        PartialTensorShape unknown({-1});
        PartialTensorShape pt = unknown.Concatenate(PartialTensorShape(input_proto));
        ShapeHandle batch_shape;
        TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(pt, &batch_shape));

        c->set_output(0, batch_shape);
        c->set_output(1, c->Scalar());
        return Status::OK();
        })
    .SetIsStateful()
    .Doc(R"doc(
    Experimental
    )doc");

  REGISTER_OP("BufferPairCompressor")
    .Attr("pack: bool = false")
    .Input("buffer_pool: Ref(string)")
    .Input("buffer_pair: string")
    .Output("compressed_buffer: string")
    .Output("time: int64")
    .Output("duration: int64")
    .Output("original_size: int32")
    .Output("compressed_size: int32")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        using namespace shape_inference;
        for (int i = 0; i < 2; i++) {
          TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        c->set_output(0, c->Vector(2));
        for (int i = 1; i < 5; i++) {
            c->set_output(i, c->Scalar());
        }
        return Status::OK();
      })
    .Doc(R"doc(
  Compresses the prepared buffer_pair records into a buffer.
  pack: pack into binary bases. will cause an error if the bufferpair does not contain bases.
  )doc");

    REGISTER_OP("BufferListCompressor")
    .Input("buffer_pool: Ref(string)")
    .Input("buffer_list: string")
    .Output("buffer: string")
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        c->set_output(0, c->input(1));
        return Status::OK();
        })
  .Doc(R"doc(
  Compresses the prepared buffer_list records and into individual buffers, and then outputs them
  )doc");

    REGISTER_OP("BufferListToBufferConverter")
            .Input("buffer_pool: Ref(string)")
            .Input("buffer_list: string")
            .Output("buffer: string")
            .SetShapeFn([](InferenceContext *c) {
                for (int i = 0; i < 2; i++) {
                    TF_RETURN_IF_ERROR(check_vector(c, i, 2));
                }
                c->set_output(0, c->input(1));
                return Status::OK();
            })
            .Doc(R"doc(
  Aggregates the prepared buffer_list records and into individual buffers, and then outputs them
  )doc");

        REGISTER_OP("BaseBufferConverter")
        .Input("num_records: int32")
        .Input("buffer: string")
        .Output("output_buffer: string")
        .SetShapeFn([](InferenceContext *c) {
        TF_RETURN_IF_ERROR(check_scalar(c, 0));
        TF_RETURN_IF_ERROR(check_vector(c, 1, 2));
        c->set_output(0, c->input(1));
        return Status::OK();
        })
        .Doc(R"doc(
Converts a base buffer to a compacted buffer
Useful after bases are used for alignment with snap
)doc");

    REGISTER_REFERENCE_POOL("RawFileSystemColumnPool")
    .Doc(R"doc(
Creates and initializes a pool containing a RawFileSystemColumn objects, in the pool as Column objects for downstream.
  )doc");

    REGISTER_OP("RawFileConverter")
    .Input("column_pool: Ref(string)")
    .Input("data: string")
    .Output("column: string")
    .Output("record_id: string")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext *c) {
        for (int i = 0; i < 2; i++) {
            TF_RETURN_IF_ERROR(check_vector(c, i, 2));
        }
        c->set_output(0, c->Vector(2));
        c->set_output(1, c->Scalar());
        return Status::OK();
    })
    .Doc(R"doc(
Converts an uncompressed file into a raw file. This op will error if you pass a compressed chunk.
The column will take ownership of the underlying data and output a Column type downstream.

This is basically if you need unstructured access to a raw file column (e.g. for the merge stage, which for now just deals in uncompressed raw files).
    )doc");

    REGISTER_OP("AtomicCounter")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("handle: resource")
    .SetIsStateful()
    .SetShapeFn(ScalarShape)
    .Doc(R"doc(
Return an atomic counter. The underlying mechanism is a c++ std::atomic
with consistency guarantees only about the actual value (not to be used as a barrier elsewhere).
    )doc");

    REGISTER_OP("AtomicCounterIncrementer")
    .Input("counter: resource")
    .Input("delta: int64")
    .SetIsStateful()
    .SetShapeFn(
            [](InferenceContext *c) {
                for (int i = 0; i < 2; i++) {
                    TF_RETURN_IF_ERROR(check_scalar(c, i));
                }
                return Status::OK();
            }
    )
    .Doc(R"doc(
Increment the atomic counter by the input amount
This is done atomically (as the name implies).
    )doc");

    REGISTER_OP("AtomicCounterFetchAndSet")
    .Input("counter: resource")
    .Input("new_value: int64")
    .SetIsStateful()
    .Output("stored_value: int64")
    .SetShapeFn(
            [](InferenceContext *c) {
                for (int i = 0; i < 2; i++) {
                    TF_RETURN_IF_ERROR(check_scalar(c, i));
                }
                c->set_output(0, c->input(1));
                return Status::OK();
            }
    )
    .Doc(R"doc(
Set the atomic counter to a provided value and retrieve its prior value.
    )doc");

    REGISTER_REFERENCE_POOL("CephLazyColumnPool")
    .Attr("cluster_name: string")
    .Attr("user_name: string")
    .Attr("pool_name: string")
    .Attr("ceph_conf_path: string")
    .Attr("records_per_segment: int >= 1")
    .Attr("num_segments: int >= 1")
    .Doc(R"doc(
Creates and initializes a pool containing a CephLazyColumn objects, in the pool as Column objects for downstream.
  )doc");

    REGISTER_OP("LazyCephReader")
            .Attr("delete_after_read: bool = false")
            .Input("column_pool: Ref(string)")
            .Input("key: string")
            .Input("namespace: string")
            .Output("column_handle: string")
            .Output("record_id: string")
    .SetIsStateful()
    .SetShapeFn(
            [](InferenceContext *c) {
                TF_RETURN_IF_ERROR(check_vector(c, 0, 2));
                for (int i = 1; i < 3; i++) {
                    TF_RETURN_IF_ERROR(check_scalar(c, i));
                }
                c->set_output(0, c->Vector(2));
                c->set_output(1, c->Scalar());
                return Status::OK();
            })
    .Doc(R"doc(
Produce lazily-read ceph segments. The asynchronous segments will have IO started in the background for each segment
upon initialization.
    )doc");

    REGISTER_OP("CephRemove")
            .Attr("cluster_name: string")
            .Attr("user_name: string")
            .Attr("pool_name: string")
            .Attr("ceph_conf_path: string")
            .Attr("columns: list(string) >= 1")
            .Input("keys: string")
            .Input("namespaces: string")
            .Output("items_deleted: int32")
            .SetShapeFn(
                    [](InferenceContext *c) {
                        ShapeHandle key_input, ns_input;
                        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &key_input));
                        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &ns_input));
                        auto k_handle = c->Dim(key_input, 0);
                        auto ns_handle = c->Dim(ns_input, 0);
                        auto k_known = c->ValueKnown(k_handle), ns_known = c->ValueKnown(ns_handle);
                        if (k_known) {
                            if (ns_known) {
                                auto k_val = c->Value(k_handle), ns_val = c->Value(ns_handle);
                                if (k_val != ns_val) {
                                    return Internal("CephRemove: ranks unknown but unequal. ", k_val, " != ", ns_val);
                                }
                            } else {
                                return Internal("CephRemove: key shape known, but namespace shape unknown");
                            }
                        } else if (ns_known) {
                            return Internal("key shape unknown in CephRemove while ns_rank is known");
                        }
                        c->set_output(0, c->Scalar());
                        return Status::OK();
                    })
    .Doc(R"doc(
Remove a bunch items from the ceph specified ceph pool
    )doc");

    REGISTER_OP("ResultsIndexCreator")
    .Input("index_pool: Ref(string)")
    .Input("column: string")
    .Output("index: string")
    .SetIsStateful()
    .SetShapeFn(
            [](InferenceContext *c) {
                TF_RETURN_IF_ERROR(check_vector(c, 0, 2));
                c->set_output(0, c->input(0));
                return Status::OK();
            }
    )
    .Doc(R"doc(
Creates a results index, used for the merge op
    )doc");

    REGISTER_REFERENCE_POOL("ResultsIndexPool")
    .Doc(R"doc(
Creates and initializes a pool containing ResultsIndices
  )doc");

    REGISTER_REFERENCE_POOL("PrimedBufferPairPool")
    .Attr("num_records: int >= 1")
    .Attr("record_size: int >= 1")
    .Doc(R"doc(
Creates and initializes a pool containing a pair of char buffers of size `buffer_size` bytes
  )doc");
}
