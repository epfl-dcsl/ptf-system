# Description:
#   Contains kernels for Barrier.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

filegroup(
    name = "cc_files",
    srcs = glob(["*.cc"]),
    visibility = ["//tensorflow:__subpackages__"],
)

# split this out to manage which headers are exposed, if necessary
filegroup(
    name = "pub_headers",
    srcs = glob(["*.h"]),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "gate_kernels",
    srcs = [":cc_files"],
    hdrs = [":pub_headers"],
    linkstatic = 1,
    deps = [
        "//tensorflow/contrib/gate/framework:gate",
    ],
    alwayslink = 1,
)
