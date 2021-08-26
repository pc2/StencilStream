#!/usr/bin/env bash

if (( $# != 1 )); then
    echo "Usage: $0 <target>" 1>&2;
    exit 1
fi

source /cm/shared/opt/intel_oneapi/2021.1.1/setvars.sh
ml compiler/GCC nalla_pcie/19.4.0

make $1