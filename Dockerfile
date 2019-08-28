FROM ubuntu:18.04 AS builder

# Dependencies to be install for all phases (run and build)
ENV RUNDEPS apt-utils software-properties-common python3-pip zlib1g-dev libboost-all-dev librados-dev libpython3-dev libs3-dev libbwa-dev libsparsehash-dev

# Dependencies to be installed only for the execution phase
ENV BUILDDEPS g++ swig bash-completion wget openjdk-8-jre openjdk-8-jdk python3-setuptools unzip

ENV BUILDTYPE "opt"
ENV BUILTOPTS "--copt=-msse4.1 --copt=-msse4.2 --copt -O3 --strip=always"

RUN apt-get update
RUN apt-get -y install --no-install-recommends $RUNDEPS $BUILDDEPS && rm -rf /var/lib/apt/lists*

RUN pip3 install --upgrade pip virtualenv setuptools wheel

RUN wget "https://github.com/bazelbuild/bazel/releases/download/0.18.1/bazel_0.18.1-linux-x86_64.deb" && apt-get -y install ./bazel* && rm ./bazel*.deb
WORKDIR /ptf
COPY . /ptf
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
RUN yes n | ./default_configure.sh
RUN bazel build $BUILDOPTS -j $(nproc) -c $BUILDTYPE //tensorflow/tools/pip_package:build_pip_package
RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package pip_pkg

FROM ubuntu:18.04 AS executor

# TODO not sure how to make this the same for both, if that's possible
ENV RUNDEPS apt-utils software-properties-common python3-pip zlib1g-dev libboost-all-dev librados-dev libpython3-dev libs3-dev libbwa-dev libsparsehash-dev
RUN apt-get update
RUN apt-get -y install --no-install-recommends $RUNDEPS && rm -rf /var/lib/apt/lists*
RUN pip3 install --upgrade pip setuptools
WORKDIR /ptf
COPY --from=builder /ptf/pip_pkg/*.whl /ptf
RUN pip3 install --trusted-host pypi.python.org *.whl
