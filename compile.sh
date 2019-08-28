#!/usr/bin/env bash

# unofficial "bash strict mode"
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
# Commented out 'u' because the `activate` script has some unassignment issues
set -u
set -eo pipefail
IFS=$'\n\t'

build_type="opt"

DIR=$(dirname $(realpath $0))

process_args() {
case "$#" in
    0)
        ;;
    1)
        build_type="$1"
        ;;
    *)
        echo "Script only accepts 0 or 1 argument!"
        exit 2
        ;;
esac
}

process_args "$@"

extra_opts=""
if [ $build_type == "opt" ]; then
    extra_opts="--copt -O3 --strip=always"
elif [ $build_type == "vtune" ]; then
    build_type="opt"
    extra_opts="--copt -g --copt -O3"
elif [ $build_type == "perf" ]; then
    build_type="dbg"
    extra_copts="--strip=never --copt -pg --copt -g --copt -O0"
elif [ $build_type == "fastperf" ]; then
    build_type="opt"
    extra_copts="--strip=never --copt -pg --copt -O0 --copt -fno-omit-frame-pointer --copt -g"
elif [ $build_type == "pprof" ]; then
    build_type="opt"
    extra_copts="--copt -pg --copt -O2 --copt -fno-omit-frame-pointer"
fi

# this is the option to build with the old ABI
# don't use because it'll cause linking issues with old libraries :/
# extra_opts="${extra_opts} --cxxopt -D_GLIBCXX_USE_CXX11_ABI=0"
sse_opts="--copt=-msse4.1 --copt=-msse4.2"

echo "Building configuration $build_type"
#max_build_threads=$(bc <<< "scale=0; ($(nproc) * 1.1) / 1" )
max_build_threads=$(nproc)
set +u
eval "bazel build $sse_opts $extra_opts -j $max_build_threads -c $build_type //tensorflow/tools/pip_package:build_pip_package"
