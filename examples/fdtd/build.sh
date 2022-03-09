#!/usr/bin/env bash
# Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

if [[ $@ == *"-h"* || $@ == *"--help"* || -z $1 ]]
then
cat <<EOF
Usage: $0 variant

This script build variants of the FDTD application. The naming scheme of the variants is as follows:

fdtd_<material_resolver>_<backend>[_report]

The placeholders may have the following values:
* material_resolver:
    * coef: Store the final material coefficients in every cell.
    * lut: Store a lookup table with all known material coefficients in the kernel and store only an index in the cell.

* backend:
    * mono: Use the monotile FPGA backend of StencilStream
    * tiling: Use the tiling FPGA backend of StencilStream
    * cpu: Use the testing CPU backend of StencilStream

The FPGA backends also support the report suffix, where a synthesis report is generated that allows
to make certain predicitions over the resulting performance.
EOF
exit 0
fi

function run_build {
    EXEC_NAME=$1
    rm -f $EXEC_NAME

    ARGS="-std=c++20 -DSTENCIL_INDEX_WIDTH=64 -DFDTD_BURST_SIZE=1024 -qactypes -I./ -O3"

    # Noctua-specific options
    if [[ -n "$EBROOTGCC" ]]
    then
        ARGS="$ARGS --gcc-toolchain=$EBROOTGCC"
    fi

    # Material resolvers
    if [[ "$EXEC_NAME" == *"coef"* ]]
    then
        ARGS="$ARGS -DCOEF_MATERIALS"
    else
        ARGS="$ARGS -DLUT_MATERIALS"
    fi

    # FPGA-specific options
    if [[ "$EXEC_NAME" == *"mono"*  || "$EXEC_NAME" == *"tiling"* ]]
    then
        ARGS="$ARGS -fintelfpga -reuse-exe=$1 -Xshardware -Xsv"

        if [[ -n $AOCL_BOARD_PACKAGE_ROOT ]]
        then
            ARGS="$ARGS -Xsboard=$FPGA_BOARD_NAME -Xsboard-package=$AOCL_BOARD_PACKAGE_ROOT"
        fi

        if [[ "$EXEC_NAME" == *"report"* ]]
        then
            ARGS="$ARGS -fsycl-link"
        fi
    fi

    if [[ "$EXEC_NAME" == *"mono"* ]]
    then
        ARGS="$ARGS -DEXECUTOR=0"
    elif [[ "$EXEC_NAME" == *"tiling"* ]]
    then
        ARGS="$ARGS -DEXECUTOR=1"
    elif [[ "$EXEC_NAME" == *"cpu"* ]]
    then
        ARGS="$ARGS -DEXECUTOR=2"
    fi

    COMMAND="dpcpp src/*.cpp -o $EXEC_NAME $ARGS"
    echo $COMMAND
    echo $COMMAND | bash
}

run_build $1
