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
#include "gate_interface.h"
namespace tensorflow {
  using namespace std;
    using namespace errors;

    namespace {
        GateInterface::IDandCountType get_top_elem(const Tensor &id_and_count, const size_t idx) {
            auto id_and_count_matrix = id_and_count.matrix<GateInterface::IDandCountType>();
            auto end_dim = id_and_count_matrix.dimension(0)-1;
            return id_and_count_matrix(end_dim, idx);
        }
    }

  string GateInterface::DebugString() {
    static string b("A gate");
    return b;
  }

    GateInterface::IDandCountType GateInterface::get_id(const Tensor &id_and_count) {
        return get_top_elem(id_and_count, 0);
    }

    GateInterface::IDandCountType GateInterface::get_count(const Tensor &id_and_count) {
        return get_top_elem(id_and_count, 1);
    }

} // namespace tensorflow {
