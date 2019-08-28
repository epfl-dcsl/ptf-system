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
from . import gate
from . import gate_runner
from tensorflow.python.framework import dtypes

"""
A bunch of modules to be used for convenience methods
"""

def _streaming_queue(id_and_count_components, parallel, batch_size=None, whole_dataset=False, name=None, shared_name=None):
    """
    :param id_and_count_components: an iterable or generator of elements with the following type (id_and_count, components)
    :param capacity: a strictly positive capacity for the gate
    :param parallel: the number of dequeuing results you want from this
    :return: a generator of gate.dequeue() results
    """
    # in case the user passes in a float or something else
    # better to catch here than an esoteric error deep down
    try:
        parallel = int(parallel)
    except TypeError as te:
        raise Exception("parallel is not an int! got {}".format(parallel))

    if parallel < 1:
        raise Exception("Dequeuing count must be at least 1. Got {}".format(parallel))
    try:
        id_and_count_components[0] # this will throw a TypeError if this is a generator
    except TypeError:
        # in case we're passed a generator
        id_and_count_components = tuple(id_and_count_components)
    if len(id_and_count_components) == 0:
        raise Exception("Must pass at least one element to enqueue into the StreamingGate")

    first_idcc = id_and_count_components[0]
    id_and_count_example, components_example = first_idcc
    component_dtypes = tuple(t.dtype for t in components_example)
    shapes = tuple(t.shape for t in components_example)

    kwargs = {}
    if name is not None and len(name) > 0:
        kwargs["name"] = name

    sg = gate.StreamingGate(
        id_and_count_upstream=id_and_count_example,
        dtypes=component_dtypes, shapes=shapes,
        limit_upstream=False, limit_downstream=False,
        shared_name=shared_name,
        **kwargs)

    enqueue_ops = [sg.enqueue(id_and_count=idc, components=components)
                   for idc, components in id_and_count_components]
    gate_runner.add_gate_runner(gate_runner=gate_runner.GateRunner(gate=sg, enqueue_ops=enqueue_ops))

    def generator(batch_size):
        if whole_dataset:
            for _ in range(parallel):
                yield sg.dequeue_whole_dataset()
        elif batch_size is None:
            for _ in range(parallel):
                yield sg.dequeue()
        else:
            try:
                batch_size = int(batch_size)
            except TypeError as te:
                raise Exception("Batch size is not an int: {}".format(batch_size))
            if batch_size < 1:
                raise Exception("Batch size must be at least 1. Got {}".format(batch_size))
            for _ in range(parallel):
                yield sg.dequeue_many(count=batch_size)
    return generator(batch_size=batch_size)

def streaming_gate(*args, **kwargs):
    return _streaming_queue(*args, **kwargs)
