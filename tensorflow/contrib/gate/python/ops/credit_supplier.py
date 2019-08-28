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

import threading
import itertools

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.contrib.gate.protobuf import gate_runner_pb2

CREDIT_SUPPLIERS = "credit_suppliers"

class CreditSupplier(object):
    _acceptable_exceptions = (errors.CancelledError,)
    def __init__(self, supply_op=None, upstream_gate=None, downstream_gate=None, close_op=None, cancel_op=None, device=None):
        if supply_op is None:
            if upstream_gate is None or downstream_gate is None:
                raise ValueError("up- and downstream gates must be specified if no supply op is provided")
            supply_op = upstream_gate.supply_downstream_credits_from_gate(downstream_gate=downstream_gate)
        elif upstream_gate is not None or downstream_gate is not None:
            raise ValueError("can't specify up or downstream gates if supply op is specified")
        self._lock = threading.Lock()
        self._supply_op = supply_op
        if close_op is None:
            if upstream_gate is None:
                raise ValueError("must use gate-style constructor if no close op is specified")
            self._close_op = upstream_gate.close(cancel_pending_enqueues=False)
        else:
            self._close_op = close_op
        if cancel_op is None:
            if upstream_gate is None:
                raise ValueError("must use gate-style constructor if no cancel op is specified")
            self._cancel_op = upstream_gate.close(cancel_pending_enqueues=True)
        else:
            self._cancel_op = cancel_op
        self._exceptions_raised = []

        if device is None:
            self._device = supply_op.device
        else:
            self._device = device

    def _run(self, sess, supply_op, coord=None):
        try:
            supply_op_callable = sess.make_callable(supply_op)
            while coord is None or not coord.should_stop():
                try:
                    supply_op_callable()
                except self._acceptable_exceptions as e:
                    try:
                        sess.run(self.close_op)
                    except Exception:
                        pass
        except Exception as e:
            with self._lock:
                self._exceptions_raised.append(e)
            if coord is not None:
                coord.request_stop(e) # give up on everything
            else:
                try:
                    sess.run(self.cancel_op)
                except Exception:
                    pass
                raise e

    def _close_on_stop(self, sess, coord):
        coord.wait_for_stop()
        try:
            with self._lock:
                if len(self._exceptions_raised) == 0:
                    op = self.close_op
                else:
                    op = self.cancel_op
            sess.run(op)
        except Exception as e:
            pass

    @property
    def smart_name(self):
        return "CreditRunner_downstream_{ds}_upstream_{us}_device_{dev}".format(ds=self.downstream_name,
                                                                                us=self.op_name, dev=self.device)

    @property
    def name(self):
        return "CreditRunner_upstream_{us}".format(us=self.op_name)

    @property
    def device(self):
        return self._device

    def create_threads(self, sess, coord=None, daemon=False, start=False):
        # try to get the name as descriptively as possible, but fall back in case something weird happens
        try:
            name = self.smart_name
        except:
            name = self.name
        credit_thread = threading.Thread(target=self._run, args=(sess, self.supply_op, coord), daemon=daemon, name=name)
        threads = [credit_thread]
        if coord is not None:
            threads.append(
                threading.Thread(target=self._close_on_stop, args=(sess, coord), daemon=daemon, name="{}_closer".format(name))
            )
            for t in threads:
                coord.register_thread(t)
        if start:
            for t in threads:
                t.start()
        return threads

    @property
    def supply_op(self):
        return self._supply_op

    @property
    def close_op(self):
        return self._close_op

    @property
    def cancel_op(self):
        return self._cancel_op

    @property
    def op_name(self):
        return self.supply_op.inputs[0].name

    @property
    def downstream_name(self):
        return self.supply_op.inputs[1].op.inputs[0].name

    def to_proto(self, export_scope=None):
        if export_scope is None or self.name.startswith(export_scope):
            credit_runner_def = gate_runner_pb2.CreditSupplierDef()
            credit_runner_def.supply_op_name = ops.strip_name_scope(
                self.supply_op.name, export_scope=export_scope
            )
            credit_runner_def.close_op_name = ops.strip_name_scope(
                self.close_op.name, export_scope=export_scope
            )
            credit_runner_def.cancel_op_name = ops.strip_name_scope(
                self.cancel_op.name, export_scope=export_scope
            )
            credit_runner_def.device = self.device
            return credit_runner_def
        else:
            return None

    @staticmethod
    def from_proto(credit_supplier_def, import_scope=None):
        assert isinstance(credit_supplier_def, gate_runner_pb2.CreditSupplierDef)
        g = ops.get_default_graph()
        supply_op = g.as_graph_element(
            ops.prepend_name_scope(credit_supplier_def.supply_op_name, import_scope=import_scope)
        )
        cancel_op = g.as_graph_element(
            ops.prepend_name_scope(credit_supplier_def.cancel_op_name, import_scope=import_scope)
        )
        close_op = g.as_graph_element(
            ops.prepend_name_scope(credit_supplier_def.close_op_name, import_scope=import_scope)
        )
        device = credit_supplier_def.device
        return CreditSupplier(supply_op=supply_op, cancel_op=cancel_op, close_op=close_op, device=device)

def add_credit_supplier(credit_supplier, collection=CREDIT_SUPPLIERS):
    ops.add_to_collection(collection, credit_supplier)

def add_credit_supplier_from_gates(upstream_gate, downstream_gate, collection=CREDIT_SUPPLIERS):
    return add_credit_supplier(credit_supplier=CreditSupplier(upstream_gate=upstream_gate,
                                                              downstream_gate=downstream_gate),
                               collection=collection)

def start_credit_suppliers(sess=None, coord=None, daemon=True, start=True, collection=CREDIT_SUPPLIERS, device=None):
    if sess is None:
        sess = ops.get_default_session()
        if sess is None:
            raise ValueError("Cannot credit suppliers. No default session is registered, and it wasn't specified")
    with sess.graph.as_default():
        return list(itertools.chain.from_iterable(cs.create_threads(
            sess=sess, coord=coord, daemon=daemon, start=start
        ) for cs in ops.get_collection(collection) if device is None or cs.device == device))

ops.register_proto_function(CREDIT_SUPPLIERS,
                            proto_type=gate_runner_pb2.CreditSupplierDef,
                            to_proto=CreditSupplier.to_proto,
                            from_proto=CreditSupplier.from_proto)
