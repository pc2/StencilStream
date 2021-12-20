#!/usr/bin/env bash
# Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

if [[ $1 == "-h" ]]
then
cat <<EOF
Usage: $0 <variant|target>

This script build variants of the FDTD application. Calling $0 <variant> builds the given variant. 
The naming scheme of the variants is as follows:

fdtd_<backend>[_report]

The placeholders may have the following values:
* executor:
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

    ARGS="-std=c++17 -DSTENCIL_INDEX_WIDTH=64 -DFDTD_BURST_SIZE=1024 -I./ -O3"

    # Noctua-specific options
    if [[ -n "$EBROOTGCC" ]]
    then
        ARGS="$ARGS --gcc-toolchain=$EBROOTGCC"
    fi

    # FPGA-specific options
    if [[ "$EXEC_NAME" == *"mono"*  || "$EXEC_NAME" == *"tiling"* ]]
    then
        ARGS="$ARGS -fintelfpga -Xshardware -Xsv"

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
        ARGS="$ARGS -DEXECUTOR=MONOTILE"
    elif [[ "$EXEC_NAME" == *"tiling"* ]]
    then
        ARGS="$ARGS -DEXECUTOR=TILING"
    elif [[ "$EXEC_NAME" == *"cpu"* ]]
    then
        ARGS="$ARGS -DEXECUTOR=CPU"
    fi

    COMMAND="dpcpp $ARGS src/*.cpp -o $EXEC_NAME"
    echo $COMMAND
    echo $COMMAND | bash
}

run_build $1