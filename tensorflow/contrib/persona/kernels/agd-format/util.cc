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
#include "util.h"
#include <cstring>
#include <cstdint>
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
  using namespace std;

  void DataResourceReleaser(ResourceContainer<Data> *data) {
    core::ScopedUnref a(data);
    {
      ResourceReleaser<Data> a1(*data);
      {
        DataReleaser dr(*data->get());
      }
    }
  }

    void ColumnResourceReleaser(ResourceContainer<Column> *data) {
        core::ScopedUnref a(data);
        {
            ResourceReleaser<Column> a1(*data);
            data->get()->Release();
        }
    }
} // namespace tensorflow {
