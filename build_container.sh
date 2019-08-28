#!/usr/bin/env bash
# unofficial "bash strict mode"
# http://redsymbol.net/articles/unofficial-bash-strict-mode/
# Commented out 'u' because the `activate` script has some unassignment issues
set -u
set -eo pipefail
DIR=$(dirname $(realpath $0))

pushd "$DIR"
docker build -t ptf-system .
popd
