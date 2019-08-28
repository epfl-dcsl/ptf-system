# Description:
#   Contains ops to build an input pipeline for tensorflow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_libs",
    "tf_custom_op_py_library",
    "tf_custom_op_library",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)

tf_custom_op_py_library(
    name = "gate_py",
    srcs = glob(["python/**/*.py"]) + ["__init__.py"],
    dso = [
        ":python/ops/_gate_ops.so",
    ],
    kernels = [
        # this isn't actually used for this op
        ":gate_op_lib",
        ":gate_kernels",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_gate_ops",
        # TODO idk how many of these we need
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:variables",
        "//tensorflow/contrib/gate/protobuf:protos_all_py",
    ],
)

tf_kernel_library(
    name = "gate_kernels",
    deps = [
        "//tensorflow/contrib/gate/kernels:gate_kernels",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "python/ops/_gate_ops.so",
    srcs = glob(["ops/*.cc"]),
    deps = [
        "//tensorflow/contrib/gate/kernels:gate_kernels",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["gate"],
)

tf_gen_op_wrapper_py(
    name = "gen_gate_ops",
    out = "python/ops/gen_gate_ops.py",
    require_shape_functions = True,
    deps = [
        ":gate_op_lib",
    ],
)

filegroup(
    name = "py_files",
    srcs = glob(["python/**/*.py"]),
)
