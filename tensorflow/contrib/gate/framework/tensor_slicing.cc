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
#include "tensor_slicing.h"

namespace tensorflow {

  namespace {

    template <DataType DT>
    Status HandleSliceToElement(const Tensor& parent, Tensor* element,
                                int64 index) {
      typedef typename EnumToDataType<DT>::Type T;
      DCHECK_NE(parent.dim_size(0), 0);
      DCHECK_GE(index, 0);
      auto parent_elems = parent.NumElements();
      auto parent_dim_size = parent.dim_size(0);
      auto elem_num_elems = element->NumElements();
      if (elem_num_elems != (parent_elems / parent_dim_size)) {
        TensorShape chip_shape = parent.shape();
        chip_shape.RemoveDim(0);
        return errors::Internal(
                "HandleSliceToElement Cannot copy slice: number of elements does not "
                        "match.  Shapes are: [element / provided]: ",
                element->shape().DebugString(), ", [parent slice / expected]: ",
                chip_shape.DebugString());
      }
      auto parent_as_matrix = parent.flat_outer_dims<T>();
      element->flat<T>() = parent_as_matrix.chip(index, 0);
      return Status::OK();
    }

    template <DataType DT>
    Status HandleElementToSlice(const Tensor& element, Tensor* parent, int index) {
      typedef typename EnumToDataType<DT>::Type T;
      DCHECK_NE(parent->dim_size(0), 0);
      DCHECK_GE(index, 0);
      auto parent_elems = parent->NumElements();
      auto parent_dim_size = parent->dim_size(0);
      auto elem_num_elems = element.NumElements();
      if (elem_num_elems != (parent_elems / parent_dim_size)) {
        TensorShape chip_shape = parent->shape();
        chip_shape.RemoveDim(0);
        return errors::Internal(
                "HandleElementToSlice Cannot copy slice: number of elements does not "
                        "match.  Shapes are: [element / expected]: ",
                element.shape().DebugString(), ", [parent slice / provided]: ",
                chip_shape.DebugString());
      }
      auto parent_as_matrix = parent->flat_outer_dims<T>();
      parent_as_matrix.chip(index, 0) = element.flat<T>();
      return Status::OK();
    }

  }  // namespace

  using namespace std;

  Status CopySliceToElement(const Tensor& parent, Tensor* element, int64 index) {
#define HANDLE_TYPE(DT)                                                   \
  if (parent.dtype() == DT) {                                             \
    TF_RETURN_IF_ERROR(HandleSliceToElement<DT>(parent, element, index)); \
    return Status::OK();                                                  \
  }
    HANDLE_TYPE(DT_FLOAT);
    HANDLE_TYPE(DT_HALF);
    HANDLE_TYPE(DT_DOUBLE);
    HANDLE_TYPE(DT_INT32);
    HANDLE_TYPE(DT_UINT8);
    HANDLE_TYPE(DT_INT16);
    HANDLE_TYPE(DT_INT8);
    HANDLE_TYPE(DT_STRING);
    HANDLE_TYPE(DT_COMPLEX64);
    HANDLE_TYPE(DT_COMPLEX128);
    HANDLE_TYPE(DT_INT64);
    HANDLE_TYPE(DT_BOOL);
    HANDLE_TYPE(DT_QINT8);
    HANDLE_TYPE(DT_QUINT8);
    HANDLE_TYPE(DT_QINT32);
    HANDLE_TYPE(DT_QINT16);
    HANDLE_TYPE(DT_QUINT16);
    HANDLE_TYPE(DT_UINT16);
#undef HANDLE_TYPE
    return errors::Unimplemented("CopySliceToElement Unhandled data type: ",
                                 parent.dtype());
  }

// Static method
  Status CopyElementToSlice(const Tensor& element, Tensor* parent, int64 index) {
#define HANDLE_TYPE(DT)                                                   \
  if (element.dtype() == DT) {                                            \
    TF_RETURN_IF_ERROR(HandleElementToSlice<DT>(element, parent, index)); \
    return Status::OK();                                                  \
  }
    HANDLE_TYPE(DT_FLOAT);
    HANDLE_TYPE(DT_HALF);
    HANDLE_TYPE(DT_DOUBLE);
    HANDLE_TYPE(DT_INT32);
    HANDLE_TYPE(DT_UINT8);
    HANDLE_TYPE(DT_INT16);
    HANDLE_TYPE(DT_INT8);
    HANDLE_TYPE(DT_STRING);
    HANDLE_TYPE(DT_COMPLEX64);
    HANDLE_TYPE(DT_COMPLEX128);
    HANDLE_TYPE(DT_INT64);
    HANDLE_TYPE(DT_BOOL);
    HANDLE_TYPE(DT_QINT8);
    HANDLE_TYPE(DT_QUINT8);
    HANDLE_TYPE(DT_QINT32);
    HANDLE_TYPE(DT_QINT16);
    HANDLE_TYPE(DT_QUINT16);
    HANDLE_TYPE(DT_UINT16);
#undef HANDLE_TYPE
    return errors::Unimplemented("CopyElementToSlice Unhandled data type: ",
                                 element.dtype());
  }

} // namespace tensorflow {
