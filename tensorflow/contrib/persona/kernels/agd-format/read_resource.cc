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
#include "read_resource.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  ReadResource::~ReadResource() {}

  Status ReadResource::split(size_t chunk, vector<BufferList*>& bl)
  {
    return Unimplemented("resource splitting not supported for this resource");
  }

  Status ReadResource::get_next_subchunk(ReadResource **rr, vector<BufferPair*>& b)
  {
    return Unimplemented("resource splitting not supported for this resource");
  }

  bool ReadResource::reset_iter() {
    return false;
  }

  size_t ReadResource::num_records() {
    return 0;
  }

  void ReadResource::release() {}

  ReadResourceReleaser::ReadResourceReleaser(ReadResource &r) : rr_(r) {}

  ReadResourceReleaser::~ReadResourceReleaser()
  {
    rr_.release();
  }
} // namespace tensorflow {
