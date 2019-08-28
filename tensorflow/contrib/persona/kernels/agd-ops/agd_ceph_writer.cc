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
#include "agd_ceph_writer.h"

#include <algorithm>
#include <chrono>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

// #define ASYNC_LARGE_WRITE

namespace tensorflow {
  using namespace std;
  using namespace errors;

  AGDCephWriterBase::AGDCephWriterBase(OpKernelConstruction *ctx) : AGDWriterBase(ctx) {
    string cluster_name, user_name, ceph_conf_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ceph_conf_path", &ceph_conf_path));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_name", &pool_name_));

    int ret = 0;
    ret = cluster_.init2(user_name.c_str(), cluster_name.c_str(), 0);
    OP_REQUIRES(ctx, ret == 0, Internal("cluster.init2 returned ", ret, " with user ", user_name,
                                        " and cluster ", cluster_name));
    ret = cluster_.conf_read_file(ceph_conf_path.c_str());
    OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster::conf_read_file for file ", ceph_conf_path, " returned error code ", ret));

    ret = cluster_.connect();
    OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster::connect returned error status ", ret));

      OP_REQUIRES(ctx, cluster_.ioctx_create(pool_name_.c_str(), io_ctx) == 0, Internal(name(), " couldn't create context for pool '", pool_name_, "'"));
  }

    AGDCephWriterBase::~AGDCephWriterBase() {
        io_ctx.close();
        cluster_.shutdown();
    }

  void AGDCephWriterBase::Compute(OpKernelContext *ctx) {
    const Tensor *key_t, *resource_t, *namespace_t;
    OP_REQUIRES_OK(ctx, ctx->input("path", &key_t));
    OP_REQUIRES_OK(ctx, ctx->input("resource_handle", &resource_t));
    OP_REQUIRES_OK(ctx, ctx->input("namespace", &namespace_t));
    auto &name_space = namespace_t->scalar<string>()();
    auto &key = key_t->scalar<string>()();
    auto resource_vec = resource_t->vec<string>();

      io_ctx.set_namespace(name_space);

    OP_REQUIRES_OK(ctx, SetHeaderValues(ctx));

    write_buf_list_.clear();

    auto status = WritePayload(ctx, resource_vec(0), resource_vec(1), key, write_buf_list_, io_ctx);
      if (not status.ok()) {
          LOG(ERROR) << name() << " failed writing ceph payload. Namespace: '" << name_space << "', key: '" << key << "'";
          OP_REQUIRES_OK(ctx, status);
      }

    OP_REQUIRES_OK(ctx, SetOutputKey(ctx, key));
  }

  Status
  AGDCephWriterBase::SendWrite(OpKernelContext *ctx, librados::bufferlist &write_buf_list, const string &key,
                                 librados::IoCtx &io_ctx) {
      write_buf_list.push_front(ceph::buffer::create_static(sizeof(header_),
                                                            reinterpret_cast<char*>(&header_)));

      auto total_bytes_to_write = write_buf_list.length();
      DCHECK_GT(total_bytes_to_write, 0); // Check that we don't ahve an empty write
  // like to why the default `osd max write size` is 90 http://docs.ceph.com/docs/master/rados/configuration/osd-config-ref/
    const auto max_bytes = 90000000u; // TODO get this from the context
      auto start = chrono::high_resolution_clock::now();
    if (total_bytes_to_write <= max_bytes) {
#ifdef ASYNC_CEPH_OPS
        this better not compile
      auto write_completion = librados::Rados::aio_create_completion();
        auto ret = io_ctx.aio_write_full(key, write_completion, write_buf_list);
        if (ret != 0) {
            return Internal(name(), ": Small Write Ceph aio_write_full returned error code: ", ret);
        }
        ret = write_completion->wait_for_complete();
        if (ret != 0) {
            return Internal(name(), ": Small Write Ceph unable to do aio_write_full completion. wait_for_complete() returned ", ret);
        }
        ret = write_completion->get_return_value();
        if (ret != 0) {
            return Internal(name(), ": Small Write Ceph wait_for_complete() ok, but return value is: ", ret);
        }
        write_completion->release();
#else
        auto ret = io_ctx.write_full(key, write_buf_list);
        if (ret != 0) {
            return Internal(name(), ": small write ceph synchronous version unable to write_full. Got error ", ret);
        }
#endif
    } else {
        auto ret = io_ctx.trunc(key, total_bytes_to_write);
        if (ret != 0) {
            return Internal(name(), ": ceph truncate for key '", key, "' with ", total_bytes_to_write, " bytes returned error ", ret);
        }

        auto remaining_bytes = total_bytes_to_write;
        uint64_t offset = 0;
#if defined(ASYNC_CEPH_OPS) && defined(ASYNC_LARGE_WRITE)
        this better not compile
        librados::AioCompletion *write_completion;
#endif
        while (remaining_bytes > 0) {
#if defined(ASYNC_CEPH_OPS) && defined(ASYNC_LARGE_WRITE)
            this better not compile
            write_completion = librados::Rados::aio_create_completion();
#endif
            const auto &bytes_to_write = min(max_bytes, remaining_bytes);
            DCHECK_GT(bytes_to_write, 0);
            DCHECK_LT(offset, total_bytes_to_write);
#if !(defined(ASYNC_CEPH_OPS) && defined(ASYNC_LARGE_WRITE))
            ret = io_ctx.write(key, write_buf_list, bytes_to_write, offset);
            if (ret != 0) {
                return Internal(name(), ": Ceph big synchronous write normal write returned error code ", ret);
            }
#else
this better not compile
            ret = io_ctx.aio_write(key, write_completion, write_buf_list, bytes_to_write, offset);
            if (ret != 0) {
                return Internal(name(), ": big write ceph aio_write() unable to be created from context. Got error code: ", ret);
            }
            ret = write_completion->wait_for_complete();
            if (ret != 0) {
                return Internal(name(), ": big write ceph wait_for_complete() erred on large write. Error code: ", ret);
            }
            ret = write_completion->get_return_value();
            if (ret != 0) {
                return Internal(name(), ": big write ceph wait_for_safe() ok, but got error code ", ret,
                ". Bytes to write: ", bytes_to_write, ", remaining: ", remaining_bytes, ", offset: ", offset, ", total: ", total_bytes_to_write);
            }
            write_completion->release();
#endif
            offset += bytes_to_write;
            remaining_bytes -= bytes_to_write;
        }
        DCHECK_EQ(remaining_bytes, 0); // shouldn't be negative
    }
      auto end = chrono::high_resolution_clock::now();
      auto write_duration = chrono::duration_cast<chrono::microseconds>(end - start);
      Tensor *stamp_t, *duration_t, *bytes_t;
      TF_RETURN_IF_ERROR(ctx->allocate_output("time", TensorShape({}), &stamp_t));
      TF_RETURN_IF_ERROR(ctx->allocate_output("duration", TensorShape({}), &duration_t));
      TF_RETURN_IF_ERROR(ctx->allocate_output("bytes", TensorShape({}), &bytes_t));
      stamp_t->scalar<int64>()() = chrono::duration_cast<chrono::microseconds>(start.time_since_epoch()).count();
      duration_t->scalar<int64>()() = write_duration.count();
      bytes_t->scalar<int32>()() = total_bytes_to_write;
  return Status::OK();
  }

} // namespace tensorflow {

