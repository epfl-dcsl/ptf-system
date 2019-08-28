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
#include <chrono>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/agd-format/data.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include <rados/librados.hpp>
#include "agd_ceph_writer.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  class CephReaderOp : public OpKernel {
  public:
    CephReaderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      string user_name, cluster_name, ceph_conf;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("read_size", &read_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_name", &pool_name_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("delete_after_read", &delete_after_read_));

      int ret = 0;
      /* Initialize the cluster handle with the "ceph" cluster name and "client.admin" user */
      ret = cluster_.init2(user_name.c_str(), cluster_name.c_str(), 0);
      OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster init2\nUsername: ", user_name, "\nCluster Name: ", cluster_name, "\nReturn code: ", ret));

      /* Read a Ceph configuration file to configure the cluster handle. */
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ceph_conf_path", &ceph_conf));
      ret = cluster_.conf_read_file(ceph_conf.c_str());
      OP_REQUIRES(ctx, ret == 0, Internal("Ceph conf file at '", ceph_conf, "' returned ", ret, " when attempting to open"));

      /* Connect to the cluster */
      ret = cluster_.connect();
      OP_REQUIRES(ctx, ret == 0, Internal("Cluster connect returned: ", ret));

        OP_REQUIRES(ctx, cluster_.ioctx_create(pool_name_.c_str(), io_ctx) == 0, Internal(name(), " unable to create io_ctx for pool '", pool_name_, "'"));
    }

    ~CephReaderOp() {
      core::ScopedUnref unref_pool(ref_pool_);
        io_ctx.close();
      cluster_.shutdown();
    }

    void Compute(OpKernelContext* ctx) override {
      if (!ref_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_pool", &ref_pool_));
      }

      const Tensor *key_t, *namespace_t;
      OP_REQUIRES_OK(ctx, ctx->input("key", &key_t));
      OP_REQUIRES_OK(ctx, ctx->input("namespace", &namespace_t));
      auto file_key = key_t->scalar<string>()();
      auto name_space = namespace_t->scalar<string>()();

      ResourceContainer<Buffer> *rec_buffer;
      OP_REQUIRES_OK(ctx, ref_pool_->GetResource(&rec_buffer));
      rec_buffer->get()->reset();

        io_ctx.set_namespace(name_space);

        auto start = chrono::high_resolution_clock::now();
      OP_REQUIRES_OK(ctx, CephReadObject(file_key, name_space, rec_buffer, io_ctx));
        auto end = chrono::high_resolution_clock::now();
        auto write_duration = chrono::duration_cast<chrono::microseconds>(end - start);
        auto total_bytes = rec_buffer->get()->size();

        //LOG(INFO) << name() << " duration: " << merge_duration.count() / 1000.0 << "ms";

      // Output tensors
      OP_REQUIRES_OK(ctx, rec_buffer->allocate_output("file_handle", ctx));

        if (delete_after_read_) {
            OP_REQUIRES(ctx, io_ctx.remove(file_key) == 0, Internal(name(), "Got error when trying to remove key '", file_key, "'"));
        }

        // LOG output here
        Tensor *stamp_t, *duration_t, *bytes_t;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("time", TensorShape({}), &stamp_t));
        OP_REQUIRES_OK(ctx, ctx->allocate_output("duration", TensorShape({}), &duration_t));
        OP_REQUIRES_OK(ctx, ctx->allocate_output("bytes", TensorShape({}), &bytes_t));
        stamp_t->scalar<int64>()() = chrono::duration_cast<chrono::microseconds>(start.time_since_epoch()).count();
        duration_t->scalar<int64>()() = write_duration.count();
        bytes_t->scalar<int32>()() = total_bytes;
    }

  private:
    long long read_size_;
      bool delete_after_read_;
    librados::Rados cluster_;
      librados::IoCtx io_ctx;
    ReferencePool<Buffer> *ref_pool_ = nullptr;
      string pool_name_;

    /* Read an object from Ceph synchronously */
    Status CephReadObject(const string &file_key, const string &name_space, ResourceContainer <Buffer> *ref_buffer,
                          librados::IoCtx &io_ctx) {
      int ret;
      auto buf = ref_buffer->get();

      size_t file_size;
      time_t pmtime;
#ifdef ASYNC_CEPH_OPS
      librados::AioCompletion *stat_completion = librados::Rados::aio_create_completion();
        ret = io_ctx.aio_stat(file_key, stat_completion, &file_size, &pmtime);
        if (ret != 0) {
            return Internal("Unable to create aio_stat completion. Error code: ", ret);
        }
        ret = stat_completion->wait_for_complete();
        if (ret != 0) {
            return Internal("CephReader: io_ctx.stat() return ", ret, " for key ", file_key);
        }
        DCHECK_EQ(ret, stat_completion->get_return_value());
        // ret = stat_completion->get_return_value();
        stat_completion->release();
#else
        ret = io_ctx.stat(file_key, &file_size, &pmtime);
        if (ret != 0) {
            return Internal(name(), ": synchronous ceph stat for key '", file_key, "', namespace '", name_space, "' returned error: ", ret);
        }
#endif

      /*char bufff[32];
      struct tm* tt = localtime(&pmtime);
      strftime(bufff, sizeof(bufff), "%b %d %H:%M", tt);
      LOG(INFO) << "Object " << file_key << " was last modified " << bufff;*/

      size_t data_read = 0;
      size_t read_len;
      size_t size_to_read = (size_t) read_size_;
      buf->resize(file_size);

      librados::bufferlist read_buf;
      while (data_read < file_size) {
        read_len = min(size_to_read, file_size - data_read);
        read_buf.push_back(ceph::buffer::create_static(read_len, &(*buf)[data_read]));

#ifdef ASYNC_CEPH_OPS
        // Create I/O Completion.
        librados::AioCompletion *read_completion = librados::Rados::aio_create_completion();
        ret = io_ctx.aio_read(file_key, read_completion, &read_buf, read_len, data_read);
        if (ret < 0) {
          return Internal(name(), ": unable to start read object. Received error ", ret);
        }
        data_read = data_read + read_len;

        // Wait for the request to complete, and check that it succeeded.
        ret = read_completion->wait_for_complete(); // complete = in memory on the OSD. safe = on desk in the OSD
        if (ret != 0) {
            return Internal(name(), " unable to wait_for_complete for read. Got code: ", ret);
        }
        ret = read_completion->get_return_value();
        if (ret < 0) {
          return Internal("Ceph Reader: unable to read object. Got error ", ret);
        }
        read_completion->release();
#else

        /* Test synchronous read */
        ret = io_ctx.read(file_key, read_buf, read_len, data_read);
        if (ret < 0) {
          return Internal(name(), ": Couldn't call io_ctx.read synchronously. Received ", ret);
        }
        data_read += read_len;
#endif
        read_buf.clear();
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("CephReader").Device(DEVICE_CPU), CephReaderOp);

} // namespace tensorflow {
