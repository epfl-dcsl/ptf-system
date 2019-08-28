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
#include "memory_region.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <sys/mman.h>
#include <sys/stat.h>
#include "tensorflow/core/lib/core/errors.h"


namespace tensorflow {

  using namespace std;
  using namespace errors;

  PosixMappedRegion::PosixMappedRegion(const void* data, uint64 size) :
    data_(data), size_(size) {}

    PosixMappedRegion::PosixMappedRegion(const void *data, uint64 size, const std::string &&file_path):
            PosixMappedRegion(data, size) {
      file_path_ = move(file_path);
    }

  PosixMappedRegion::~PosixMappedRegion() {
    if (data_) {
      munmap(const_cast<void*>(data_), size_);
    }
    if (not file_path_.empty()) {
       auto result = remove(file_path_.c_str());
      if (result != 0) {
        LOG(ERROR) << "Unable to delete file at path '" << file_path_ << "'. remove() returned " << result;
          char buf[2048];
          auto ret = strerror_r(result, buf, sizeof(buf));
        if (ret != 0) {
          LOG(ERROR) << "Got bad value on strerror(" << result << "): " << ret;
        } else {
          buf[sizeof(buf)-1] = '\0';
          LOG(ERROR) << "Got the following error for the bad deletion: " << buf;
        }
      }
    }
  }

  const void* PosixMappedRegion::data() {
    return data_;
  }

  uint64 PosixMappedRegion::length() {
    return size_;
  }

  Status PosixMappedRegion::fromFile(const string &filepath, const FileSystem &fs,
                                     unique_ptr<ReadOnlyMemoryRegion> &result, bool synchronous, bool delete_on_free) {
    string translated_fname = fs.TranslateName(filepath);
    Status s = Status::OK();
    int fd = open(translated_fname.c_str(), O_RDONLY);
    if (fd < 0) {
      s = Internal("PosixMappedRegion: file open ", filepath, " (translated: ", translated_fname, ") failed with errno ", errno);
    } else {
      struct stat st;
      ::fstat(fd, &st);
      const void* address =
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE | (synchronous ? MAP_POPULATE : 0), fd, 0);
      if (address == MAP_FAILED) {
        s = Internal("PosixMappedRegion: mmap ", filepath, " (translated: ", translated_fname, ") failed with errno ", errno);
      } else {
          if (delete_on_free) {
            result.reset(new PosixMappedRegion(address, st.st_size, move(translated_fname)));
          } else {
            result.reset(new PosixMappedRegion(address, st.st_size));
          }
      }
      close(fd);
    }
    return s;
  }

} // namespace tensorflow {
