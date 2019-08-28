# Copyright 2019 École Polytechnique Fédérale de Lausanne. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import enum
logging.basicConfig()
log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)

from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

from tensorflow.contrib.gate.python.ops import gen_gate_ops
# pylint: disable=wildcard-import
from tensorflow.contrib.gate.python.ops.gen_gate_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.framework import dtypes, tensor_shape, ops
from tensorflow.python.ops import data_flow_ops, math_ops
from tensorflow.python.summary import summary

# Need to keep this noop in in order to get it to load the shared object
# with the kernel definitions
_gate_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile('_gate_ops.so'))

class Gate(object):
    class EnqueueType(enum.Enum):
        SCALAR = "scalar"
        MANY = "many"

    class DequeueType(enum.Enum):
        SCALAR = "scalar"
        MANY = "many"
        PARTITION = "partition"

    _enqueue = staticmethod(gen_gate_ops.gate_enqueue)
    _enqueue_many = staticmethod(gen_gate_ops.gate_enqueue_many)
    _close = staticmethod(gen_gate_ops.gate_close)
    _dequeue = staticmethod(gen_gate_ops.gate_dequeue)
    _dequeue_many = staticmethod(gen_gate_ops.gate_dequeue_many)
    _dequeue_partition = staticmethod(gen_gate_ops.gate_dequeue_partition)
    _num_rounds = staticmethod(gen_gate_ops.num_rounds)
    _open_requests = staticmethod(gen_gate_ops.num_open_requests)
    _upstream_release_credits = staticmethod(gen_gate_ops.release_credits)
    _downstream_supply_credits = staticmethod(gen_gate_ops.supply_credits)

    def __init__(self, gate_type, capacity=dtypes.int32.max, id_and_count_size=None, id_and_count_upstream=None, dtypes=None, shapes=None, sample_tensors=None, sample_tensors_are_batch=False, shared_name=None, name="gate",
                 limit_upstream=True, limit_downstream=True, join=False):
        if sample_tensors is None:
            if dtypes is None or shapes is None:
                raise Exception("Either dtypes/shapes must be specified, or sample tensor given")
        elif dtypes is not None or shapes is not None:
            raise Exception("Either dtypes/shapes must be specified, or sample tensor given")
        else:
            as_tensors = [ops.convert_to_tensor(s) for s in sample_tensors]
            dtypes = [a.dtype for a in as_tensors]
            if sample_tensors_are_batch:
                shapes = [a.shape[1:] for a in as_tensors]
            else:
                shapes = [a.shape for a in as_tensors]

        if len(dtypes) != len(shapes):
            raise ValueError("Queue shapes has length {q} while dtypes has len {d}. They must be equal!".format(
                q=len(shapes), d=len(dtypes)
            ))

        if id_and_count_size is None:
            if id_and_count_upstream is None:
                raise Exception("Must specify id_and_count_size, or give an example")
            if not isinstance(id_and_count_upstream, (ops.Tensor,)):
                id_and_count_upstream = ops.convert_to_tensor(id_and_count_upstream)
            idc_shape = id_and_count_upstream.shape
            if idc_shape.ndims != 2:
                raise Exception("Expected valid matrix for id_and_count_shape example, but got ndims {}".format(idc_shape.ndims))
            id_and_count_size = idc_shape[0]
        elif id_and_count_upstream is not None:
            raise Exception("Only id_and_count_size or upstream can be specified, not both")

        if capacity < 1:
            raise Exception("Capacity must be > 0, but got {}".format(capacity))

        self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
        self._dtypes = data_flow_ops._as_type_list(dtypes=dtypes)
        self._dequeue_type = None
        self._dequeue_size = None
        self._enqueue_type = None
        self._enqueue_size = None
        self._limit_downstream = limit_downstream
        self._limit_upstream = limit_upstream
        self._id_and_count_size = id_and_count_size
        self._id_and_count_size_out = id_and_count_size
        self._capacity = capacity
        self._join = join
        if join:
            self._id_and_count_size -= 1
            self._id_and_count_size_out -= 1
            if self._id_and_count_size < 1:
                raise ValueError("Incoming id_and_count size is <1 with join type: {}".format(self._id_and_count_size))
            if self._id_and_count_size_out < 1:
                raise ValueError("Outgoing id_and_count size is <1 with join type: {}".format(self._id_and_count_size_out))
        self._gate_ref = self._make_get_type(dtypes=self._dtypes, shapes=self._shapes, capacity=capacity,
                                             limit_upstream=limit_upstream, limit_downstream=limit_downstream,
                                             gate_type=gate_type, shared_name=shared_name, name=name)
        self.make_summaries()

    def make_summaries(self):
        summary.scalar(
            name="gate/{name}/num_open_requests_max_{capacity}".format(name=self.name, capacity=self.capacity),
            tensor=self.open_requests
        )
        if self.capacity < dtypes.int32.max:
            summary.scalar(
                name="gate/{name}/open_requests_fraction_of_{capacity}_full".format(name=self.name, capacity=self.capacity),
                tensor=math_ops.cast(self.open_requests, dtype=dtypes.float32) * (1. / self.capacity)
            )
        else:
            log.info("Skipping {} because capacity is infinite".format(self.name))
        summary.scalar(
            name="gate/{name}/num_rounds".format(name=self.name),
            tensor=self.num_rounds
        )

    def _make_get_type(self, dtypes, shapes, capacity, limit_upstream, limit_downstream, gate_type, shared_name, name):
        return gate_type(
            component_types=dtypes,
            component_shapes=shapes,
            capacity=capacity,
            limit_upstream=limit_upstream,
            limit_downstream=limit_downstream,
            shared_name=shared_name, name=name)

    def enqueue(self, id_and_count, components, name="gate_enqueue"):
        if self._enqueue is None:
            raise NotImplementedError("{} doesn't support Enqueue op".format(self.name))
        if self._enqueue_type is None:
            self._enqueue_type = self.EnqueueType.SCALAR
        elif self._enqueue_type != self.EnqueueType.SCALAR:
            raise Exception("Enqueue type already set to non-scalar type. All enqueue operations must be of the same type!")
        # just in case it's a single value (for convenience)
        if not isinstance(components, (list, tuple)):
            components = (components,)
        if len(components) != len(self.dtypes):
            raise Exception("Enqueuing components has {e} count, but must have {a} count".format(
                e=len(components), a=len(self.dtypes)
            ))

        id_and_count_shape = id_and_count.shape
        if id_and_count_shape.ndims != 2:
            raise Exception("Bad number of dimensions for id_and_count. Expected 2, but got {}".format(id_and_count_shape.ndims))

        if self.is_join_gate:
            dim0_value = id_and_count_shape.dims[0].value
            if dim0_value < 1:
                raise Exception("Got size of <1 ({}) in first dimension for join gate!".format(dim0_value))
            id_and_count = id_and_count[:-1]
            id_and_count_shape = id_and_count.shape

        if id_and_count_shape.dims[0].value != self._id_and_count_size:
            raise Exception("Bad shape for enqueue op: {shape}. Expected {exp} in the 0th dimension, but got {actual}".format(
                shape=id_and_count_shape, exp=self._id_and_count_size, actual=id_and_count_shape.dims[0].value
            ))

        tensor_components = []
        for i, (val, dtype, shape) in enumerate(zip(components, self.dtypes, self.shapes)):
            tensor_component = ops.convert_to_tensor(val, dtype=dtype, name="gate_enq_component_{}".format(i))
            actual_shape = tensor_component.shape
            if actual_shape != shape:
                raise Exception("Enqueue component {i}: got shape {act}, but expected shape {exp}".format(
                    i=i, act=actual_shape, exp=shape
                ))
            tensor_components.append(tensor_component)

        return self._enqueue(
            handle=self._gate_ref,
            id_and_count=id_and_count,
            components=tensor_components,
            name=name
        )

    def enqueue_many(self, id_and_count, components, name="gate_enqueue_many"):
        """
        This should only really be used to enqueue a batch from a partition gate.

        This expects that the count for id_and_count reflects the total number of elements in the batch.

        For example, if a request has 10 elements and a batch size of 4, this op expects a batches of size 4,4,2
        (though they will all be unspecified in the 0th dimension).

        The count on all of these batches should be 10, not 3 (e.g. from a dequeue_many op).

        :param id_and_count:
        :param components:
        :return:
        """
        if self._enqueue_many is None:
            raise NotImplementedError("{} doesn't support EnqueueMany".format(self.name))
        if self.is_join_gate:
            raise NotImplementedError("{} doesn't support EnqueueMany on a join style gate".format(self.name))
        try:
            components = tuple(components)
        except TypeError:
            raise Exception("enqueue_many must have a tuple of components. Got {}".format(components))
        if len(components) != len(self.dtypes):
            raise Exception("Enqueuing components has {e} count, but must have {a} count".format(
                e=len(components), a=len(self.dtypes)
            ))

        id_and_count_shape = id_and_count.shape
        if not (id_and_count_shape.ndims == 2 and id_and_count_shape.dims[0].value == self._id_and_count_size):
            raise Exception("Bad shape for enqueue_many op: {}".format(id_and_count_shape))

        tensor_components = []
        expected_first_dim = None
        for i, (val, dtype, shape) in enumerate(zip(components, self.dtypes, self.shapes)):
            tensor_component = ops.convert_to_tensor(val, dtype=dtype, name="gate_enq_component_{}".format(i))
            actual_shape = tensor_component.shape

            expected_num_elements = shape.ndims + 1 # striped across the 0th dim
            actual_num_elements = actual_shape.ndims
            if expected_num_elements != actual_num_elements:
                raise Exception("EnqueueMany component {i}: expected {e} dims, but got {a} dims".format(
                    i=i, e=expected_num_elements, a=actual_num_elements
                ))

            batch_size = actual_shape[0]
            if expected_first_dim is None:
                expected_first_dim = batch_size
            elif expected_first_dim != batch_size:
                raise Exception("EnqueueMany: Got batch size mismatch. Expected {e} 0th dim, but got {a} for component {i}".format(
                    i=i, e=expected_first_dim, a=batch_size
                ))

            expected_shape = tensor_shape.TensorShape(actual_shape[0]).concatenate(shape)

            if actual_shape != expected_shape:
                raise Exception("EnqueueMany component {i}: got shape {act}, but expected shape {exp}".format(
                    i=i, act=actual_shape, exp=expected_shape
                ))
            tensor_components.append(tensor_component)
        if self._enqueue_type is None:
            self._enqueue_type = self.EnqueueType.MANY
            self._enqueue_size = expected_first_dim
        elif not (self._enqueue_type == self.EnqueueType.MANY and (self._enqueue_size == expected_first_dim or (self._enqueue_size.value is None and expected_first_dim.value is None))):
            raise Exception("Enqueue type is not as expected. Expected first dimension {exp}, but got {actual}".format(
                exp=self._enqueue_type, actual=expected_first_dim
            ))

        return self._enqueue_many(
            handle=self._gate_ref,
            id_and_count=id_and_count,
            components=tensor_components,
            name=name
        )

    def _set_dequeue_type_and_count(self, dequeue_type, dequeue_count):
        # both of these should already be checked by the caller's logic
        assert isinstance(dequeue_type, self.DequeueType)
        dequeue_count = int(dequeue_count)
        assert dequeue_count > 0
        if self._dequeue_type is None:
            assert self._dequeue_size is None
            self._dequeue_type = dequeue_type
            self._dequeue_size = dequeue_count
        else:
            assert self._dequeue_size is not None
            assert self._dequeue_type is not None
            if self._dequeue_type != dequeue_type:
                raise Exception("Attempting to overwrite existing dequeue type '{existing}' with '{new_type}'".format(
                    existing=self._dequeue_type, new_type=dequeue_type
                ))
            if self._dequeue_size != dequeue_count:
                raise Exception("Attempting to overwrite existing dequeue count '{existing}' with '{new_type}'".format(
                    existing=self._dequeue_size, new_type=dequeue_count
                ))

    def dequeue(self, name="gate_dequeue"):
        if self._dequeue is None:
            raise NotImplementedError("{} doesn't support the Dequeue op".format(self.name))
        self._set_dequeue_type_and_count(dequeue_type=self.DequeueType.SCALAR,
                                         dequeue_count=1)
        return self._dequeue(
            handle=self._gate_ref,
            Tcomponents=self.dtypes,
            Tshapes=self.shapes,
            id_and_count_size=self._id_and_count_size_out,
            name=name
        )

    def dequeue_many(self, count, name="gate_dequeue_many"):
        """
        All dequeue_many ops are unspecified in the 0th dimension. The count on all dimensions in the id_and_count stack are divided by count
        and rounded UP round_up(count / batch_size)

        It is the responsibility of whatever op is consuming this to collapse the unspecified batch size to 1.

        :param count:
        :return:
        """
        if self._dequeue_many is None:
            raise NotImplementedError("{} doesn't support the DequeueMany op".format(self.name))
        self._set_dequeue_type_and_count(dequeue_type=self.DequeueType.MANY,
                                         dequeue_count=count)
        return self._dequeue_many(
            handle=self._gate_ref,
            Tcomponents=self.dtypes,
            Tshapes=self.shapes,
            batch_size=count,
            id_and_count_size=self._id_and_count_size_out,
            name=name
        )

    def dequeue_partition(self, count, name="gate_dequeue_partition"):
        """
        This dequeues an entire partition. Similar to dequeue many, except the
        id_and_count is pushed with a new entry, which is [some assigned id, count of actual number of elements in the partition]

        A downstream gate should call "enqueue_many" into it, so that the downstream gate can split the operation.

        :param count: the maximum size of the partition
        :return:
        """
        if self._dequeue_partition is None:
            raise NotImplementedError("{} doesn't support the DequeuePartition op".format(self.name))
        self._set_dequeue_type_and_count(dequeue_type=self.DequeueType.PARTITION,
                                         dequeue_count=count)
        return self._dequeue_partition(
            handle=self._gate_ref,
            Tcomponents=self.dtypes,
            Tshapes=self.shapes,
            batch_size=count,
            id_and_count_size=self._id_and_count_size_out+1,
            name=name
        )

    def dequeue_whole_partition(self, name="gate_dequeue_whole_partition"):
        return self.dequeue_partition(count=dtypes.int32.max-1, name=name)

    def dequeue_whole_dataset(self, name="gate_dequeue_whole_dataset"):
        return self.dequeue_many(count=dtypes.int32.max-1, name=name)

    def close(self, cancel_pending_enqueues=False, name="gate_close"):
        return self._close(
            handle=self._gate_ref,
            cancel_pending_enqueues=cancel_pending_enqueues,
            name=name
        )

    def release_credits(self, name="release_credits"):
        """
        Release credits to an upstream gate
        :return: an op that will supply credits when they become available
        """
        if self._upstream_release_credits is None:
            raise NotImplementedError("{} doesn't support supplying upstream credits".format(self.name))
        elif not self.limit_upstream:
            raise Exception("Gate '{name}' doesn't release upstream credits".format(name=self.name))
        return self._upstream_release_credits(
            handle=self.gate_ref,
            name="{}_{}".format(self.name, name)
        )

    def supply_downstream_credits(self, downstream_credits, name="supply_credits"):
        """
        Supply credits FROM the downstream gate to this gate
        :param downstream_credits:
        :return:
        """
        if self._downstream_supply_credits is None:
            raise NotImplementedError("{} doesn't support the supply of downstream credits".format(self.name))
        elif not self.limit_downstream:
            raise Exception("Gate '{name}' doesn't limit downstream credits from downstream gate".format(name=self.name))
        return self._downstream_supply_credits(
            credits=downstream_credits,
            handle=self.gate_ref,
            name="{}_{}".format(self.name, name)
        )

    def supply_downstream_credits_from_gate(self, downstream_gate, name="credit_transfer"):
        assert isinstance(downstream_gate, Gate)
        if not self.limit_downstream:
            raise Exception("Gate '{up_name}' does not limit the downstream credits for rounds to '{down_name}' for credit supplying".format(
                down_name=downstream_gate.name, up_name=self.name
            ))
        elif not downstream_gate.limit_upstream:
            raise Exception("Gate '{down_name}' does not limit the credits to upstream gate '{up_name}'".format(
                up_name=self.name,  down_name=downstream_gate.name
            ))
        name = "{}_{}".format(self.name, name)
        return self.supply_downstream_credits(downstream_credits=downstream_gate.release_credits(name=name), name=name)

    @property
    def gate_ref(self):
        """The underlying gate reference."""
        return self._gate_ref

    @property
    def device(self):
        return self.gate_ref.device

    @property
    def name(self):
        """The name of the underlying gate."""
        return self.gate_ref.op.name

    @property
    def dtypes(self):
        """The list of dtypes for each component of a gate element."""
        return self._dtypes

    @property
    def shapes(self):
        """The list of shapes for each component of a gate element."""
        return self._shapes

    @property
    def count_size_in(self):
        return self._id_and_count_size

    @property
    def count_size_out(self):
        return self._id_and_count_size_out

    @property
    def capacity(self):
        return self._capacity

    @property
    def is_join_gate(self):
        return self._join

    @property
    def num_rounds(self):
        return self._num_rounds(self.gate_ref)

    @property
    def open_requests(self):
        return self._open_requests(self.gate_ref)

    @property
    def limit_downstream(self):
        return self._limit_downstream

    @property
    def limit_upstream(self):
        return self._limit_upstream

class StreamingGate(Gate):
    def __init__(self, name="streaming_gate", *args, **kwargs):
        super().__init__(*args, **kwargs, gate_type=gen_gate_ops.streaming_gate, name=name)

class EgressGate(Gate):
    _enqueue = staticmethod(gen_gate_ops.gate_egress_enqueue)
    _enqueue_many = staticmethod(gen_gate_ops.gate_egress_enqueue_many)
    _close = staticmethod(gen_gate_ops.gate_egress_close)
    _upstream_release_credits = staticmethod(gen_gate_ops.egress_release_credits)
    _num_rounds = staticmethod(gen_gate_ops.egress_num_rounds)
    _open_requests = staticmethod(gen_gate_ops.egress_num_open_requests)
    _dequeue = None
    _dequeue_many = None
    _dequeue_partition = None
    _downstream_supply_credits = None

    def __init__(self, name="egress_gate", *args, **kwargs):
        # capacity_key = "capacity"
        # if capacity_key in kwargs:
        #     log.warning("Egress gate got capacity specification of {}. Ignoring...".format(kwargs[capacity_key]))
        #     del kwargs[capacity_key]
        super().__init__(*args, **kwargs, gate_type=gen_gate_ops.egress_gate, name=name)
        # self._capacity = dtypes.int32.max

    def _make_get_type(self, dtypes, shapes, capacity, limit_upstream, gate_type, shared_name, name, **kwargs):
        """
        Same as parent method, except Egress gate cannot limit downstream
        :param dtypes:
        :param shapes:
        :param capacity:
        :param limit_upstream:
        :param limit_downstream:
        :param gate_type:
        :param shared_name:
        :param name:
        :return:
        """
        return gate_type(
            component_types=dtypes,
            component_shapes=shapes,
            capacity=capacity,
            limit_upstream=limit_upstream,
            shared_name=shared_name, name=name)

    def dequeue_request(self, request_id, name="gate_dequeue_request"):
        return gen_gate_ops.egress_dequeue(handle=self.gate_ref,
                                           Tcomponents=self.dtypes,
                                           Tshapes=self.shapes,
                                           requested_dataset_id=request_id,
                                           name=name)

class IngressGate(Gate):
    _dequeue = staticmethod(gen_gate_ops.gate_ingress_dequeue)
    _dequeue_many = staticmethod(gen_gate_ops.gate_ingress_dequeue_many)
    _dequeue_partition = staticmethod(gen_gate_ops.gate_ingress_dequeue_partition)
    _close = staticmethod(gen_gate_ops.gate_ingress_close)
    _downstream_supply_credits = staticmethod(gen_gate_ops.ingress_supply_credits)
    _num_rounds = staticmethod(gen_gate_ops.ingress_num_rounds)
    _open_requests = staticmethod(gen_gate_ops.ingress_num_open_requests)
    _enqueue = None
    _upstream_release_credits = None
    _enqueue_many = None

    def __init__(self, name="ingress_gate", *args, **kwargs):
        super().__init__(*args, **kwargs, gate_type=gen_gate_ops.ingress_gate, id_and_count_size=1, name=name)

    def enqueue_request(self, components, name="gate_enqueue_requeust"):
        if len(components) != len(self.shapes):
            raise Exception("components has {c} elements while shapes has {s}".format(
                c=len(components), s=len(self.shapes)
            ))
        components = [ops.convert_to_tensor(v) for v in components]
        c_dtypes = [c.dtype for c in components]
        if c_dtypes != self.dtypes:
            raise Exception("Got dtypes {a}, but expected {e}".format(a=c_dtypes, e=self.dtypes))
        cshapes = [c.shape for c in components]
        first_dim_sizes = []
        for component_shape, expected_shape in zip(cshapes, self.shapes):
            ndims = component_shape.ndims
            if ndims < 1:
                raise Exception("Shape {} is a scalar!".format(component_shape))
            first_dim_sizes.append(component_shape.dims[0])
            sliced_dims = tensor_shape.TensorShape(component_shape.dims[1:])
            if sliced_dims != expected_shape:
                raise Exception("Expected shape {e} doesn't match actual shape {a}".format(e=expected_shape,
                                                                                           a=sliced_dims))
        if first_dim_sizes[0].value is None:
            if not all(b.value is None for b in first_dim_sizes):
                raise Exception("Expected all shapes to be undefinied in the first dimension, but got dims: {}".format(first_dim_sizes))
        elif not all(b.value is not None and b == first_dim_sizes[0] for b in first_dim_sizes):
            raise Exception("Expected all shapes to have same size in first dimension, but got dims: {}".format(first_dim_sizes))

        return gen_gate_ops.ingress_enqueue(handle=self.gate_ref,
                                            components=components,
                                            Tshapes=cshapes, name=name)
