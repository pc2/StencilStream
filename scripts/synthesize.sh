#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter -p normal -q fpgasynthesis -t 1-00:00:00 -c 8 --mem 120G

if [ $# -ne 4 ]; then
    echo "Usage: $0 <package registry url> <tag name> <application> <target>" 1>&2
    exit 1
fi
PACKAGE_REGISTRY_URL=$1
TAG_NAME=$2
APPLICATION=$3
TARGET=$4

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. || exit 1
make $TARGET || exit 1
cd examples/$APPLICATION
tar -caf $TARGET.tar.gz $TARGET $TARGET.prj/reports || exit 1
curl --header "JOB-TOKEN: ${CI_JOB_TOKEN}" --upload-file ${TARGET}.tar.gz "${PACKAGE_REGISTRY_URL}/${TARGET}/${TAG_NAME}/${TARGET}.tar.gz" || exit 1