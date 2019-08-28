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
#include <vector>
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

  namespace {
      void release_completion(librados::AioCompletion *ac) {
          if (ac != nullptr) {
              ac->release();
          }
      }
  }

  class CephRemoveOp : public OpKernel {
  public:
    explicit CephRemoveOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      string user_name, cluster_name, ceph_conf;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_name", &pool_name_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("columns", &columns_));
        for (auto &c : columns_) {
            if (c[0] != '.') {
                c = "."+c;
            }
        }

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

    ~CephRemoveOp() {
        io_ctx.close();
      cluster_.shutdown();
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor *key_t, *namespace_t;
      OP_REQUIRES_OK(ctx, ctx->input("keys", &key_t));
      OP_REQUIRES_OK(ctx, ctx->input("namespaces", &namespace_t));
      auto file_keys = key_t->vec<string>();
      auto name_spaces = namespace_t->vec<string>();
      OP_REQUIRES(ctx, file_keys.dimension(0) == name_spaces.dimension(0), Internal("namespaces and file keys not of equal dimension"));

        auto num_items = file_keys.dimension(0);
        OP_REQUIRES(ctx, num_items > 0, Internal("Got empty vector of items to delete"));
        bool fail = false;
        {
            vector<unique_ptr<librados::AioCompletion, decltype(release_completion)*>> completions;
            completions.reserve(num_items*columns_.size());
            for (size_t i = 0; i < num_items; ++i) {
                auto &k = file_keys(i);
                auto &ns = name_spaces(i);
                io_ctx.set_namespace(ns);
                for (auto &ext : columns_) {
                    librados::AioCompletion *stat_completion = librados::Rados::aio_create_completion();
                    OP_REQUIRES(ctx, stat_completion != nullptr, Internal("Couldn't create a rados completion"));
                    completions.emplace_back(stat_completion, release_completion);
                    auto final_key_name = k+ext;
                    auto ret = io_ctx.aio_remove(final_key_name, stat_completion);
                    OP_REQUIRES(ctx, ret == 0, Internal("aio_remove creation returned an error"));
                }
            }
            for (auto &c : completions) {
                c->wait_for_complete();
            }
        }

        OP_REQUIRES(ctx, not fail, Internal("One or more completions failed"));

        Tensor *t;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{}, &t));
        t->scalar<int32>()() = num_items;
    }

  private:
    librados::Rados cluster_;
      librados::IoCtx io_ctx;
      string pool_name_;
    vector<string> columns_;
  };

  REGISTER_KERNEL_BUILDER(Name("CephRemove").Device(DEVICE_CPU), CephRemoveOp);

} // namespace tensorflow {
