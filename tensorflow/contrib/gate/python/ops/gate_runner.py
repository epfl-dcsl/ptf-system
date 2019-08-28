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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import weakref
import threading
import itertools

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.contrib.gate.protobuf import gate_runner_pb2
from . import gate as ggate

GATE_RUNNERS = "gate_runners"

class GateRunner(object):
    def __init__(self, gate, enqueue_ops, cancel_op=None, close_op=None, gate_closed_exception_types=None, device=None, request_stop_on_success=False):
        # TODO have more asserts here to protect the incoming types
        self._gate = gate
        self._enqueue_ops = enqueue_ops
        assert len(enqueue_ops) > 0

        # used for accounting / tracking
        self._enqueue_ops_per_session = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

        # TODO allow these to be specified
        if gate_closed_exception_types is None:
            exception_types = (errors.OutOfRangeError,)
        elif isinstance(gate_closed_exception_types, (list, tuple)):
            exception_types = gate_closed_exception_types
        else:
            exception_types = (gate_closed_exception_types,)

        for e in exception_types:
            if not issubclass(e, errors.OpError):
                raise Exception("Can't except non-{} type for gate exceptions: {}".format(errors.OpError, e))

        self._gate_closed_exception_types = exception_types
        assert isinstance(request_stop_on_success, bool)
        self._request_stop_on_success = request_stop_on_success

        self._exceptions_raised = []

        if close_op is None:
            self._close_op = self._gate.close(cancel_pending_enqueues=False)
        else:
            self._close_op = close_op

        if cancel_op is None:
            self._cancel_op = self._gate.close(cancel_pending_enqueues=True)
        else:
            self._cancel_op = cancel_op
        if device is None:
            all_devices = { enq_op_input.device for enq_op_input in itertools.chain.from_iterable(
                enq_op.inputs[1:] for enq_op in enqueue_ops
            ) }
            if len(all_devices) > 1:
                raise Exception("Have more than 1 device for inputs. Please specify this manually for constructing this gate_runner.\nGot: {}".format(all_devices))
            assert len(all_devices) == 1
            device = all_devices.pop()
        self._device = device

    def _run(self, sess, enqueue_op, coord=None):
        """
        If coord is None, then this stop coordinator is unused
        :param sess:
        :param enqueue_op:
        :param coord:
        :return:
        """
        decremented = False
        try:
            enqueue_callable = sess.make_callable(enqueue_op)
            while coord is None or not coord.should_stop():
                try:
                    enqueue_callable()
                except self._gate_closed_exception_types as e:
                    with self._lock:
                        self._enqueue_ops_per_session[sess] -= 1
                        decremented = True
                        if self._enqueue_ops_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception:
                                pass
                            finally:
                                if self._request_stop_on_success:
                                    if coord is None:
                                        print("Requesting stop on success not possible! {name} doesn't have a coordinator".format(name=self.name))
                                    else:
                                        coord.request_stop()
                    return # to break out of the loop
        except Exception as e:
            if coord is not None:
                coord.request_stop(e)
            else:
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            if not decremented:
                with self._lock:
                    self._enqueue_ops_per_session[sess] -= 1

    def _close_on_stop(self, sess, coord):
        coord.wait_for_stop()
        try:
            with self._lock:
                if len(self._exceptions_raised) == 0:
                    sess.run(self._close_op)
                else:
                    sess.run(self._cancel_op)
        except Exception as e:
            pass # TODO log this somehow

    def create_threads(self, sess, coord=None, daemon=False, start=False):
        # set up and create all the create_threads stuff, as well as the
        # prior stuff
        with self._lock:
            try:
                if self._enqueue_ops_per_session[sess] > 0:
                    return []
            except KeyError:
                pass
            self._enqueue_ops_per_session[sess] = len(self._enqueue_ops)
            self._exceptions_raised.clear() # yes, we use python3

        ret_threads = [threading.Thread(target=self._run, args=(sess, e, coord), name="{}_gate_runner_enqueuer_device_{}".format(self.name, self.device), daemon=daemon) for e in self._enqueue_ops]
        if coord is not None:
            ret_threads.append(threading.Thread(target=self._close_on_stop, args=(sess, coord), daemon=daemon, name="{}_gate_runner_coordinator_device_{}".format(self.name, self.device)))
        for t in ret_threads:
            if coord is not None:
                coord.register_thread(t)
            if start:
                t.start()
        return ret_threads

    @property
    def gate(self):
        return self._gate

    @property
    def device(self):
        return self._device

    @property
    def enqueue_ops(self):
        return self._enqueue_ops

    @property
    def close_op(self):
        return self._close_op

    @property
    def cancel_op(self):
        return self._cancel_op

    @property
    def gate_closed_exception_types(self):
        return self._gate_closed_exception_types

    @property
    def name(self):
        """The string name of the underlying Queue."""
        return self._gate.name

    @property
    def smart_name(self):
        return "{}_device_{}".format(self.name, self.device)

    def to_proto(self, export_scope=None):
        if (export_scope is None or self.gate.name.startswith(export_scope)):
            gate_runner_def = gate_runner_pb2.GateRunnerDef()
            gate_runner_def.gate_name = ops.strip_name_scope(
                self.gate.name, export_scope
            )
            for enqueue_op in self.enqueue_ops:
                gate_runner_def.enqueue_op_name.append(
                    ops.strip_name_scope(enqueue_op.name, export_scope)
                )
            gate_runner_def.close_op_name = ops.strip_name_scope(
                self.cancel_op.name, export_scope
            )
            gate_runner_def.cancel_op_name = ops.strip_name_scope(
                self.close_op.name, export_scope
            )
            gate_runner_def.device = self.device
            gate_runner_def.gate_closed_exception_types.extend(
                errors.error_code_from_exception_type(cls)
                for cls in self.gate_closed_exception_types
            )
            gate_runner_def.request_stop_on_success = self._request_stop_on_success
            return gate_runner_def
        else:
            return None

    @staticmethod
    def from_proto(gate_runner_def, import_scope=None):
        assert isinstance(gate_runner_def, gate_runner_pb2.GateRunnerDef)
        g = ops.get_default_graph()
        gate = g.as_graph_element(
            ops.prepend_name_scope(gate_runner_def.gate_name, import_scope))
        enqueue_ops = [
            g.as_graph_element(ops.prepend_name_scope(op, import_scope)) for op in gate_runner_def.enqueue_op_name
        ]
        close_op = g.as_graph_element(ops.prepend_name_scope(gate_runner_def.close_op_name, import_scope))
        cancel_op = g.as_graph_element(ops.prepend_name_scope(gate_runner_def.cancel_op_name, import_scope))
        device = gate_runner_def.device
        gate_closed_exception_types = tuple(
           errors.exception_type_from_error_code(code)
           for code in gate_runner_def.gate_closed_exception_types)
        if len(gate_closed_exception_types) == 0:
            gate_closed_exception_types = (errors.OutOfRangeError,)
        request_stop_on_success = gate_runner_def.request_stop_on_success
        return GateRunner(gate=gate,
                          device=device,
                          enqueue_ops=enqueue_ops,
                          cancel_op=cancel_op,
                          close_op=close_op,
                          gate_closed_exception_types=gate_closed_exception_types,
                          request_stop_on_success=request_stop_on_success)

def add_gate_runner(gate_runner, collection=GATE_RUNNERS):
    ops.add_to_collection(collection, gate_runner)

def gate_runner(gate, ops, collection=GATE_RUNNERS):
    add_gate_runner(
        gate_runner=GateRunner(gate=gate, enqueue_ops=ops),
        collection=collection
    )

def start_gate_runners(sess=None, coord=None, daemon=True, start=True, collection=GATE_RUNNERS, device=None):
    if sess is None:
        sess = ops.get_default_session()
        if not sess:
            raise ValueError("Cannot start gate runners. No default session is registered, and it wasn't specified")
    with sess.graph.as_default():
        return list(
            itertools.chain.from_iterable(
                gate_runner.create_threads(sess=sess, coord=coord, daemon=daemon, start=start)
                for gate_runner in ops.get_collection(collection) if device is None or gate_runner.device == device
            )
        )

ops.register_proto_function(GATE_RUNNERS,
                            proto_type=gate_runner_pb2.GateRunnerDef,
                            to_proto=GateRunner.to_proto,
                            from_proto=GateRunner.from_proto)