#!/usr/bin/env bash
# Copyright © 2020-2022 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
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

This script builds variants of the FDTD application. The application was many different ways to do
things, which are selected via macro constants and templates throughout the code. The actual
combinations one might want to build are expressed via the passed executable name. The scheme of 
the variant names is:

fdtd_<material_resolver>_<backend>[_<modifier>]

The `material_resolver` denotes how the material of a cell is stored in a cell and how the 
material coefficients are retrieved from it. Possible values are:
* `coef`: Store the final material coefficients directly in every cell.
* `lut`: Store a lookup table with all known material coefficients in the kernel and store only an index in the cell.

StencilStream offers different backends or executors with different architectures or goals. The possible values are:
* `mono`: Use the monotile FPGA backend of StencilStream. It yields a higher performance for the 
  same pipeline length than `tiling`, but it is limited to a maximal grid width and height.
* `tiling`: Use the tiling FPGA backend of StencilStream. It can handle arbitrarly large grids (and
  therefore cavity radii and resolutions), but generally yields a lower performance than `mono`.
* `cpu`: Use the testing CPU backend of StencilStream. This backend is a trivial implementation of
  executor interface for CPUs and therefore performs worse than both FPGA backends when synthesized,
  but it's good enough to complete most simulations in reasonable times for testing purposes.

The FPGA backends also support some modifiers:
* `emu`: Don't synthesize the design and create an emulation image. Note that emulation is *very* 
  slow and not suitable to test the functionality of the transition function. Use the `cpu` backend 
  instead.
* `report`: Generate are hardware usage report for the variant. These reports can deliver a good 
  estimate on the synthesized design's performance.
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
        ARGS="$ARGS -DMATERIAL=0"
    elif [[ "$EXEC_NAME" == *"lut"* ]]
    then
        ARGS="$ARGS -DMATERIAL=1"
    fi

    # FPGA-specific options
    if [[ "$EXEC_NAME" == *"mono"*  || "$EXEC_NAME" == *"tiling"* ]]
    then
        ARGS="$ARGS -fintelfpga -Xsv"

        if [[ "$EXEC_NAME" != *"emu"* ]]
        then
            ARGS="$ARGS -DHARDWARE -reuse-exe=$1 -Xshardware"
            
            if [[ -n $AOCL_BOARD_PACKAGE_ROOT ]]
            then
                ARGS="$ARGS -Xsboard=$FPGA_BOARD_NAME -Xsboard-package=$AOCL_BOARD_PACKAGE_ROOT"
            fi

            if [[ "$EXEC_NAME" == *"report"* ]]
            then
                ARGS="$ARGS -fsycl-link"
            fi
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
