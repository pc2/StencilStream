#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter -p normal -t 00:15:00 -c 8 --mem 4G

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
cmake -DCMAKE_BUILD_TYPE=Release ..
make $TARGET
cd examples/$APPLICATION
tar -caf $TARGET.tar.gz $TARGET
curl --header "JOB-TOKEN: ${CI_JOB_TOKEN}" --upload-file ${TARGET}.tar.gz "${PACKAGE_REGISTRY_URL}/${TARGET}/${TAG_NAME}/${TARGET}.tar.gz"