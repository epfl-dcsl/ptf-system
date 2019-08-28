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
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
    using namespace std;
    using namespace errors;

    namespace {
        const string op_name("LogEvents");
    }

    class LogEventsOp : public OpKernel {
    public:
        explicit LogEventsOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
            vector<string> item_names;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("item_names", &item_names));
            string directory, my_name;
            OP_REQUIRES_OK(ctx, ctx->GetAttr("directory", &directory));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("event_name", &my_name));

            struct stat info;
            if (stat(directory.c_str(), &info) != 0) {
                OP_REQUIRES_OK(ctx, Internal("Couldn't stat directory at path '", directory, "'"));
            } else if ((info.st_mode & S_IFDIR) == 0) {
                OP_REQUIRES_OK(ctx, Internal("Path '", directory, "' is not a directory"));
            }
            string sanitized_name = name();
            for (auto &c : sanitized_name) {
                if (c == '/') {
                    c = ':';
                }
            }

            string file_path = directory + "/" + my_name + "_" + sanitized_name + "_event_log.csv";

            outfile_.rdbuf()->pubsetbuf(0,0);
            outfile_.open(file_path);
            OP_REQUIRES(ctx, outfile_.good(), Internal("Unable to open output file at '", file_path, "'"));

            auto num_items = item_names.size();
            for (size_t i = 0; i < num_items; ++i) {
                outfile_ << item_names[i];
                if (i+1 < num_items) {
                    outfile_ << ",";
                }
            }
            outfile_ << "\n";
        }

        void Compute(OpKernelContext *ctx) override {
            OpInputList components;
            OP_REQUIRES_OK(ctx, ctx->input_list("components", &components));
            auto num_comp = components.size();
            for (decltype(num_comp) i = 0; i < num_comp; ++i) {
                auto const& comp = components[i];
                outfile_ << comp.SummarizeValue(INT64_MAX);
                if (i+1 < num_comp) {
                    outfile_ << ",";
                }
            }
            outfile_ << "\n";
        }
    private:
        ofstream outfile_;
    };
    REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), LogEventsOp);
}
