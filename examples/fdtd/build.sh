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
Usage: $0 <variant|target>

This script build variants of the FDTD application. Calling $0 <variant> builds the given variant. 
The naming scheme of the variants is as follows:

fdtd_<material_resolver>_<architecture>_<target>

The placeholders may have the following values:
* architecture:
    * mono: Use the monotile backend of StencilStream
    * tiling: Use the tiling backend of StencilStream

* target: 
    * emu: Compile the device code for emulation
    * hw: Compile the devie code for hardware execution
    * report: Analyse the device code and generate the synthesis report

* material_resolver:
    * coef: Store the final material coefficients in every cell.
    * lut: Store a lookup table with all known material coefficients in the kernel and store only an index in the cell.

Alternatively, you can also just give a target instead of a variant. In this case, $0 will build all
variants for the given target.
EOF
exit 0
fi

function run_build {
    EXEC_NAME=$1
    rm -f $EXEC_NAME

    ARGS="-fintelfpga -std=c++17 -DSTENCIL_INDEX_WIDTH=64 -DFDTD_BURST_SIZE=1024 -I./ -O3"

    if [[ -n "$EBROOTGCC" ]]
    then
        ARGS="$ARGS --gcc-toolchain=$EBROOTGCC"
    fi

    if [[ "$EXEC_NAME" == *"mono"* ]]
    then
        ARGS="$ARGS -DMONOTILE"
    fi

    if [[ "$EXEC_NAME" == *"coef"* ]]
    then
        ARGS="$ARGS -DCOEF_MATERIALS"
    else
        ARGS="$ARGS -DLUT_MATERIALS"
    fi

    if [[ "$EXEC_NAME" == *"hw"* || "$EXEC_NAME" == *"report"* ]]
    then
        ARGS="$ARGS -DHARDWARE -Xshardware -Xsv -Xsprofile"

        if [[ -n $AOCL_BOARD_PACKAGE_ROOT ]]
        then
            ARGS="$ARGS -Xsboard=$FPGA_BOARD_NAME -Xsboard-package=$AOCL_BOARD_PACKAGE_ROOT"
        fi

        if [[ "$EXEC_NAME" == *"report"* ]]
        then
            ARGS="$ARGS -fsycl-link"
        fi
    fi

    COMMAND="dpcpp $ARGS src/*.cpp -o $EXEC_NAME"
    echo $COMMAND
    echo $COMMAND | bash
}

if [[ $1 == "emu" ]]
then
    SUFFIX="emu"
fi

if [[ $1 == "hw" ]]
then
    SUFFIX="hw"
fi

if [[ $1 == "report" ]]
then
    SUFFIX="report"
fi

if [[ -n $SUFFIX ]]
then
    for NAME in "fdtd_coef_mono_" "fdtd_lut_mono_" "fdtd_coef_tiling_" "fdtd_lut_tiling_"
    do
        run_build "$NAME$SUFFIX"
    done
else
    run_build $1
fi