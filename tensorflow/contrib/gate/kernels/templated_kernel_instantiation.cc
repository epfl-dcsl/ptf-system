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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/gate/framework/gate_interface.h"
#include "tensorflow/contrib/gate/framework/egress_gate.h"
#include "tensorflow/contrib/gate/framework/ingress_gate.h"
#include "close.h"
#include "enqueue.h"
#include "dequeue.h"
#include "dequeue_many.h"
#include "enqueue_many.h"
#include "dequeue_partition.h"
#include "request_credits.h"
#include "supply_credits.h"
#include "gate_stats.h"

#define SYSTEM_NAME "Gate"

namespace tensorflow {

#define MAKE_OP(_NAME_, _INTERFACE_, _OP_) \
    REGISTER_KERNEL_BUILDER(Name(SYSTEM_NAME _NAME_).Device(DEVICE_CPU), _OP_<_INTERFACE_>)

#define CLOSE_OP(_NAME_, _INTERFACE_) \
    MAKE_OP(_NAME_ "Close", _INTERFACE_, GateCloseOp)

#define ENQUEUE_OP(_NAME_, _INTERFACE_) \
    MAKE_OP(_NAME_ "Enqueue", _INTERFACE_, GateEnqueueOp)

#define ENQUEUE_MANY_OP(_NAME_, _INTERFACE_) \
    MAKE_OP(_NAME_ "EnqueueMany", _INTERFACE_, GateEnqueueManyOp)

#define DEQUEUE_OP(_NAME_, _INTERFACE_) \
    MAKE_OP(_NAME_ "Dequeue", _INTERFACE_, GateDequeueOp)

#define DEQUEUE_MANY_OP(_NAME_, _INTERFACE_) \
    MAKE_OP(_NAME_ "DequeueMany", _INTERFACE_, GateDequeueManyOp)

#define DEQUEUE_PARTITION_OP(_NAME_, _INTERFACE_) \
    MAKE_OP(_NAME_ "DequeuePartition", _INTERFACE_, GateDequeuePartitionOp)

#define TEMPLATED_CLOSE_AND_ENQUEUE(_NAME_, _INTERFACE_) \
  CLOSE_OP(_NAME_, _INTERFACE_); \
  ENQUEUE_OP(_NAME_, _INTERFACE_); \
  ENQUEUE_MANY_OP(_NAME_, _INTERFACE_)

#define TEMPLATED_DEQUEUE_OPS(_NAME_, _INTERFACE_) \
    DEQUEUE_OP(_NAME_, _INTERFACE_); \
    DEQUEUE_MANY_OP(_NAME_, _INTERFACE_); \
    DEQUEUE_PARTITION_OP(_NAME_, _INTERFACE_)

#define UPSTREAM_RELEASE_CREDITS_OP(_NAME_, _INTERFACE_) \
    REGISTER_KERNEL_BUILDER(Name(_NAME_ "ReleaseCredits").Device(DEVICE_CPU), GateRequestCreditsOp<_INTERFACE_>)

#define DOWNSTREAM_SUPPLY_CREDITS_OP(_NAME_, _INTERFACE_) \
    REGISTER_KERNEL_BUILDER(Name(_NAME_ "SupplyCredits").Device(DEVICE_CPU), GateSupplyCreditsOp<_INTERFACE_>)

#define BOTH_CREDIT_OPS(_NAME_, _INTERFACE_) \
    UPSTREAM_RELEASE_CREDITS_OP(_NAME_, _INTERFACE_); \
    DOWNSTREAM_SUPPLY_CREDITS_OP(_NAME_, _INTERFACE_)

#define BOTH_STATS_OPS(_NAME_, _INTERFACE_) \
    REGISTER_KERNEL_BUILDER(Name(_NAME_ "NumOpenRequests").Device(DEVICE_CPU), NumOpenRequestsOp<_INTERFACE_>); \
    REGISTER_KERNEL_BUILDER(Name(_NAME_ "NumRounds").Device(DEVICE_CPU), NumRoundsOp<_INTERFACE_>)

  TEMPLATED_CLOSE_AND_ENQUEUE("", GateInterface);
  TEMPLATED_CLOSE_AND_ENQUEUE("Egress", EgressGate);

  CLOSE_OP("Ingress", IngressGate);

    TEMPLATED_DEQUEUE_OPS("", GateInterface);
    TEMPLATED_DEQUEUE_OPS("Ingress", IngressGate);

    BOTH_CREDIT_OPS("", GateInterface);
    UPSTREAM_RELEASE_CREDITS_OP("Egress", EgressGate);
    DOWNSTREAM_SUPPLY_CREDITS_OP("Ingress", IngressGate);

    BOTH_STATS_OPS("", GateInterface);
    BOTH_STATS_OPS("Ingress", IngressGate);
    BOTH_STATS_OPS("Egress", EgressGate);

} // namespace tensorflow {
