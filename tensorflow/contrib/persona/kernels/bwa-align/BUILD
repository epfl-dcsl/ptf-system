load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_library(
    name = "bwa_lib",
    #srcs = glob(["libsnap.a"]),
    srcs = glob(
        ["bwa/*.c"],
        exclude = ["bwa/main*"],
    ),
    hdrs = glob(["bwa/*.h"]),
    includes = ["bwa"],
    linkopts = [
        "-lm",
        "-lpthread",
        "-lz",
        "-lrt",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
    #copts = ["-pthread", "-MMD", "-msse", "-mssse3", "-msse4.2", "-Wno-format", "-std=c++11"]
)

cc_library(
    name = "bwa_align",
    srcs = glob(["*.cc"]),
    hdrs = ["bwa_wrapper.h"],
    copts = tf_copts() + ["-fexceptions"],
    includes = ["."],
    linkstatic = 1,
    nocopts = "-fno-exceptions",
    visibility = ["//visibility:public"],
    deps = [
        ":bwa_lib",
        "//tensorflow/contrib/persona/kernels/agd-format:agd_format",
        "//tensorflow/contrib/persona/kernels/concurrent_queue",
        "//tensorflow/contrib/persona/kernels/object-pool:object_pool",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@protobuf_archive//:protobuf",
    ],
    alwayslink = 1,
)
