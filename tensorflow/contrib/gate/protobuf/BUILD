licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

filegroup(
    name = "proto_files",
    srcs = glob(["*.proto"]),
    visibility = ["//tensorflow:__subpackages__"],
)

# TODO this sometimes has issues?
tf_proto_library(
    name = "protos_all",
    srcs = ["gate_runner.proto"],
    visibility = ["//visibility:public"],
)
