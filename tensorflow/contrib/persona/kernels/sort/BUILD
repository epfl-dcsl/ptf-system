# sorting operations

load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "agd_sort",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts() + ["-fexceptions"],
    linkopts = ["-lrados"],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/gate/framework:gate",
        "//tensorflow/contrib/persona/kernels/agd-format:agd_format",
    ],
)
