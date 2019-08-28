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
#include <string>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

  using namespace errors;
  using namespace std;
  using namespace shape_inference;

  // has to be a macro so that you can use string pasting of adjacent literals
#define SYSTEM_NAME "Gate"

  inline Status check_vector(InferenceContext *c, size_t input_idx, size_t dim_size) {
    ShapeHandle input_data;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(input_idx), 1, &input_data));
    auto dim_handle = c->Dim(input_data, 0);
    auto dim_value = c->Value(dim_handle);
    if (dim_value != dim_size) {
      return Internal("Op expected tensor of size ", dim_size, ", but got ", dim_value);
    }
    return Status::OK();
  }

  inline Status check_scalar(InferenceContext *c, size_t input_idx) {
    ShapeHandle input_data;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(input_idx), 0, &input_data));
    return Status::OK();
  }

  inline Status check_two_matrix(InferenceContext *c, size_t input_idx) {
    ShapeHandle id_and_count;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(input_idx), 2, &id_and_count));
    auto dim_value = c->Value(c->Dim(id_and_count, 1));
    if (dim_value != 2) {
      return InvalidArgument("Got an invalid dim-1 for id_and_count matrix: ", dim_value);
    }
    return Status::OK();
  }

    // Pulled out so egress, which doesn't have a capacity, can also use this
  Status check_gate_common(InferenceContext *c) {
      vector<DataType> dtypes;
      vector<TensorShape> shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("component_shapes", &shapes));
      TF_RETURN_IF_ERROR(c->GetAttr("component_types", &dtypes));
      auto dtypes_size = dtypes.size();
      auto shapes_size = shapes.size();
      if (dtypes_size != shapes_size) {
          return errors::InvalidArgument("shapes and dtypes must have equal size");
      } else if (dtypes_size == 0) {
          return errors::InvalidArgument("must specify at least one tensor for dataset gate");
      }
      c->set_output(0, c->Scalar());
      return Status::OK();
  }

    Status check_gate_with_capacity(InferenceContext *c) {
        TF_RETURN_IF_ERROR(check_gate_common(c));
        int32 capacity;
        TF_RETURN_IF_ERROR(c->GetAttr("capacity", &capacity));
        if (capacity < 1) {
            return errors::InvalidArgument("capacity must be strictly > 0, but got ", capacity);
        }
        c->set_output(0, c->Scalar());
        return Status::OK();
    }

#define UPSTREAM_RELEASE_CREDITS_OP(_NAME_) \
    REGISTER_OP(_NAME_ "ReleaseCredits") \
    .Input("handle: resource") \
    .Output("credits: int32")

#define SUPPLY_DOWNSTREAM_CREDITS_OP(_NAME_) \
    REGISTER_OP(_NAME_ "SupplyCredits") \
    .Input("handle: resource") \
    .Input("credits: int32")

#define BOTH_CREDIT_OPS(_NAME_) \
    UPSTREAM_RELEASE_CREDITS_OP(_NAME_); \
    SUPPLY_DOWNSTREAM_CREDITS_OP(_NAME_)

#define GATE_COMMON_ATTRIBUTES \
            .Output("handle: resource") \
            .Attr("component_types: list(type) >= 1") \
            .Attr("component_shapes: list(shape) >= 0 = []") \
            .Attr("container: string = ''") \
            .Attr("shared_name: string = ''") \
            .SetIsStateful()

#define UPSTREAM_LIMIT \
    .Attr("limit_upstream: bool = true")

#define DOWNSTREAM_LIMIT \
    .Attr("limit_downstream: bool = true")

#define CAPACITY_LIMIT \
    .Attr((string("capacity: int = ")+to_string(kint32max-1)).c_str())

#define GATE_OP(GATE_NAME)  \
    REGISTER_OP(GATE_NAME SYSTEM_NAME) \
            GATE_COMMON_ATTRIBUTES \
            UPSTREAM_LIMIT \
            CAPACITY_LIMIT \
            DOWNSTREAM_LIMIT \
            .SetShapeFn(check_gate_with_capacity)

#define GATE_SCALAR_OP(GATE_NAME, OP_NAME, ATTR_STRING) \
    REGISTER_OP(GATE_NAME OP_NAME) \
        .Input("handle: resource") \
        .Output(ATTR_STRING ": int32") \
        .SetIsStateful() \
        .SetShapeFn([](InferenceContext *c) { \
            c->set_output(0, c->Scalar()); \
            return Status::OK(); \
        })

#define NUM_OPEN_REQUESTS_OP(GATE_NAME) \
    GATE_SCALAR_OP(GATE_NAME, "NumOpenRequests", "open_requests")

#define NUM_ROUNDS_OP(GATE_NAME) \
    GATE_SCALAR_OP(GATE_NAME, "NumRounds", "num_rounds")

#define STATS_OPS(GATE_NAME) \
    NUM_OPEN_REQUESTS_OP(GATE_NAME); \
    NUM_ROUNDS_OP(GATE_NAME)

    STATS_OPS("");
    STATS_OPS("Ingress");
    STATS_OPS("Egress");

    GATE_OP("Streaming")
    .Doc(R"doc(
A dataset gate that allows elements to stream through.
Unlike the BatchingGate, it doesn't queue up elements until they have all arrived before releasing them.
This is the one you probably want to use, for pipelining.

Note that there is no guarantee on ordering for processing epochs in parallel (e.g. not by alphabetical order by ID).
All operations and epochs with this gate are treated independently.

Requests for enqueue and dequeue are still processed in-order.

Capacity is the number of open requests that may exist in this system at any given time.
)doc");
    BOTH_CREDIT_OPS("");

    REGISTER_OP("Egress" SYSTEM_NAME)
    GATE_COMMON_ATTRIBUTES
    UPSTREAM_LIMIT
    CAPACITY_LIMIT
    .SetShapeFn(check_gate_common)
  .Doc(R"doc(
A gate that will return only the requested id results, once they are available,
all in a single batch.

See the associated dequeue op.

This is most useful for egress from a pipeline, i.e. a client waiting for all the results to complete.

This gate is not bounded in size.
)doc");
    UPSTREAM_RELEASE_CREDITS_OP("Egress");

GATE_OP("Ingress")
  .Attr("id_start : int = 0")
  .Doc(R"doc(
A gate that will enqueue a single dataset as a whole.
See the associated enqueue op.

This is most useful for an ingress pipeline.
)doc");
    SUPPLY_DOWNSTREAM_CREDITS_OP("Ingress");

  GATE_OP("Partition")
  .Doc(R"doc(
A gate that has the capability to dequeue many as a bunch of single ops

This is most useful for provided batching across parallelism
)doc");

#define ENQUEUE_OP(_NAME_) \
  REGISTER_OP(SYSTEM_NAME _NAME_ "Enqueue") \
          .Input("handle: resource") \
          .Input("id_and_count: int32") \
          .Input("components: Tcomponents") \
          .SetShapeFn([](InferenceContext* c) { \
            return check_two_matrix(c, 1); \
          }) \
          .Attr("Tcomponents: list(type) >= 1") \
  .Doc(R"doc( \
)doc")

#define ENQUEUE_MANY_OP(_NAME_) \
  REGISTER_OP(SYSTEM_NAME _NAME_ "EnqueueMany") \
          .Input("handle: resource") \
          .Input("id_and_count: int32") \
          .Input("components: Tcomponents") \
          .SetShapeFn([](InferenceContext* c) { \
            return check_two_matrix(c, 1); \
          }) \
          .Attr("Tcomponents: list(type) >= 1") \
  .Doc(R"doc( \
)doc")

#define CLOSE_OP(_NAME_) \
  REGISTER_OP(SYSTEM_NAME _NAME_ "Close") \
          .Input("handle: resource") \
          .Attr("cancel_pending_enqueues: bool = false") \
          .Doc(R"doc( \
)doc")

#define TEMPLATED_GATE_OP(_NAME_) \
  ENQUEUE_OP(_NAME_); \
  CLOSE_OP(_NAME_); \
  ENQUEUE_MANY_OP(_NAME_)

  // Note that Push type can reuse streaming for all operations
  // because all of its only slightly modified operation
  TEMPLATED_GATE_OP("");
  TEMPLATED_GATE_OP("Egress");

  CLOSE_OP("Ingress"); // Ingress doesn't have common enqueue

    Status dequeue_op_shape_fn(InferenceContext *c) {
        vector<DataType> t_components;
        vector<TensorShape> t_shapes;
        int id_and_count_dims;
        TF_RETURN_IF_ERROR(c->GetAttr("Tcomponents", &t_components));
        TF_RETURN_IF_ERROR(c->GetAttr("Tshapes", &t_shapes));
        TF_RETURN_IF_ERROR(c->GetAttr("id_and_count_size", &id_and_count_dims));
        auto t_comp_size = t_components.size();
        auto t_shape_size = t_shapes.size();
        if (t_comp_size != t_shape_size) {
            return errors::InvalidArgument("Tcomponents has ", t_comp_size,
                                           " elements while Tshapes has ", t_shape_size);
        }
        vector<ShapeHandle> components;
        ShapeHandle h;
        for (const auto &shape : t_shapes) {
            TF_RETURN_IF_ERROR(c->MakeShapeFromTensorShape(shape, &h));
            components.push_back(h);
        }
        c->set_output(0, c->Matrix(id_and_count_dims, 2));
        c->set_output("components", components);
        return Status::OK();
    }

#define DEQUEUE_OP(_NAME_)  \
REGISTER_OP(SYSTEM_NAME _NAME_ "Dequeue") \
          .Input("handle: resource") \
          .Output("id_and_count: int32") \
          .Output("components: Tcomponents") \
          .SetShapeFn(dequeue_op_shape_fn) \
          .Attr("Tcomponents: list(type) >= 1") \
          .Attr("Tshapes: list(shape) >= 1") \
          .Attr("id_and_count_size: int >= 1") \
          .Doc(R"doc( \
)doc")

    Status dequeue_many_op_shape_fn(InferenceContext *c) {
        vector<DataType> t_components;
        vector<PartialTensorShape> t_shapes;
        int id_and_count_dims;
        TF_RETURN_IF_ERROR(c->GetAttr("Tcomponents", &t_components));
        TF_RETURN_IF_ERROR(c->GetAttr("Tshapes", &t_shapes));
        TF_RETURN_IF_ERROR(c->GetAttr("id_and_count_size", &id_and_count_dims));
        auto t_comp_size = t_components.size();
        auto t_shape_size = t_shapes.size();
        if (t_comp_size != t_shape_size) {
            return errors::InvalidArgument("Tcomponents has ", t_comp_size,
                                           " elements while Tshapes has ", t_shape_size);
        }

        vector<ShapeHandle> components;
        ShapeHandle h;
        for (auto &shape : t_shapes) {
            shape.InsertDim(0, -1);
            TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &h));
            components.push_back(h);
        }
        c->set_output("components", components);

        c->set_output(0, c->Matrix(id_and_count_dims, 2));
        return Status::OK();
    }

#define DEQUEUE_BULK_OP(_NAME_) \
  REGISTER_OP(_NAME_) \
          .Input("handle: resource") \
          .Output("id_and_count: int32") \
          .Output("components: Tcomponents") \
          .SetShapeFn(dequeue_many_op_shape_fn) \
          .Attr("Tcomponents: list(type) >= 1") \
          .Attr("Tshapes: list(shape) >= 1") \
          .Attr("batch_size: int >= 1") \
          .Attr("id_and_count_size: int >= 1") \
          .Doc(R"doc( \
)doc");

#define DEQUEUE_MANY_OP(_NAME_) DEQUEUE_BULK_OP(SYSTEM_NAME _NAME_ "DequeueMany")
#define DEQUEUE_PARTITION_OP(_NAME_) DEQUEUE_BULK_OP(SYSTEM_NAME _NAME_ "DequeuePartition")

#define DEQUEUE_OPS(_NAME_) \
    DEQUEUE_OP(_NAME_); \
    DEQUEUE_MANY_OP(_NAME_); \
    DEQUEUE_PARTITION_OP(_NAME_)

    DEQUEUE_OPS("");
    DEQUEUE_OPS("Ingress");

  REGISTER_OP("EgressDequeue")
          .Input("handle: resource")
          .Input("requested_dataset_id: int32")
          .Output("components: Tcomponents")
          .SetShapeFn([](InferenceContext* c) {
            TF_RETURN_IF_ERROR(check_scalar(c, 1));
            vector<DataType> t_components;
            vector<PartialTensorShape> t_shapes;
            TF_RETURN_IF_ERROR(c->GetAttr("Tcomponents", &t_components));
            TF_RETURN_IF_ERROR(c->GetAttr("Tshapes", &t_shapes));
            auto t_comp_size = t_components.size();
            auto t_shape_size = t_shapes.size();
            if (t_comp_size != t_shape_size) {
              return errors::InvalidArgument("Tcomponents has ", t_comp_size,
                                             " elements while Tshapes has ", t_shape_size);
            }

            vector<ShapeHandle> components;
            ShapeHandle h;
            for (auto &shape : t_shapes) {
              shape.InsertDim(0, -1);
              TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &h));
              components.push_back(h);
            }
            c->set_output("components", components);
            return Status::OK();
          })
          .Attr("Tcomponents: list(type) >= 1")
          .Attr("Tshapes: list(shape) >= 1")
          .Doc(R"doc(
This is just like the normal dequeue, except count will be 1 (so it's not outputted).
Get everything for the requested_dataset_id in one fell swoop.
)doc");

    REGISTER_OP("IngressEnqueue")
    .Input("handle: resource")
    .Input("components: Tcomponents")
    .Output("id_and_count: int32")
    .SetShapeFn([](InferenceContext *c) {
        vector<DataType> t_components;
        vector<PartialTensorShape> t_shapes;
        TF_RETURN_IF_ERROR(c->GetAttr("Tcomponents", &t_components));
        TF_RETURN_IF_ERROR(c->GetAttr("Tshapes", &t_shapes));
        auto t_comp_size = t_components.size();
        auto t_shape_size = t_shapes.size();
        if (t_comp_size != t_shape_size) {
            return errors::InvalidArgument("Tcomponents has ", t_comp_size,
                                           " elements while Tshapes has ", t_shape_size);
        }

        int64 dim_size = -1, elem_dim_size;
        for (auto &shape : t_shapes) {
            if (shape.dims() < 1) {
                return errors::InvalidArgument("No scalar arguments allowed for ingress enqueue. must be at least a vector");
            }
            elem_dim_size = shape.dim_size(0);
            if (dim_size == -1) {
                dim_size = elem_dim_size;
            } else if (dim_size != elem_dim_size) {
                return errors::InvalidArgument("One of the components doesn't have the same shape in the 0th dimension");
            }
        }
        c->set_output(0, c->Scalar());
        return Status::OK();
    })
    .Attr("Tcomponents: list(type) >= 1")
    .Attr("Tshapes: list(shape) >= 1")
    .Doc(R"doc(
)doc");

#undef SYSTEM_NAME

    REGISTER_OP("UnixTimestamp")
    .Output("timestamp: int64")
    .SetIsStateful() // So that the runtime doesn't get cute and cache this value
    .SetShapeFn([](InferenceContext *c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
    })
    .Doc(R"doc(
Return the current time in Unix Time in SECONDS since the epoch
)doc");

    REGISTER_OP("LogEvents")
    .SetIsStateful()
    .Attr("item_names: list(string)")
    .Attr("directory: string")
    .Attr("event_name: string")
    .Attr("Tcomponents: list(type) >= 1")
    .Input("components: Tcomponents")
            .SetShapeFn([](InferenceContext *c) {
                vector<DataType> types;
                TF_RETURN_IF_ERROR(c->GetAttr("Tcomponents", &types));
                vector<string> names;
                TF_RETURN_IF_ERROR(c->GetAttr("item_names", &names));
                auto t_size = types.size(), n_size = names.size();
                if (t_size != n_size) {
                    return InvalidArgument("LogEvents: Tcomponents size (", t_size, ") != item names size (", n_size, ")");
                }
                return Status::OK();
            })
    .Doc(R"doc(
    )doc");

} // namespace tensorflow {
