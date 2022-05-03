#!/usr/bin/env bash
# Copyright © 2020-2022 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

if [[ $@ == *"-h"* || $@ == *"--help"* || $# < 3 ]]
then
cat <<EOF
Usage: $0 material_resolver source time_resolver backend [modifier]

This script builds variants of the FDTD application. The application has many different ways to do
things, which are selected via macro constants and templates throughout the code. The actual
combinations one might want to build are expressed set using the CLI arguments.

The 'material_resolver' denotes how the material of a cell is stored in a cell and how the 
material coefficients are retrieved from it. Possible values are:
* 'coef': Store the final material coefficients directly in every cell.
* 'lut': Store a lookup table with all known material coefficients in the kernel and store only and 
    index in the cell.
* 'render': Use a lookup table like with `lut`, but pick the material depending on the cell's
    position. No material information is stored in the cells.

The 'source' denotes whether the computations of the source wave amplitude are done by the FPGA or
the host. If the wave is computed on the host, the amplitudes are simply stored in a look-up table. 
Possible values are:
* 'od': Compute the source wave amplitudes on-demand with the FPGA.
* 'lut': Compute the source wave amplitudes on the host and store them in a lookup table.

The 'time_resolver' denotes whether the computations of the current time is done by the FPGA or the
host. If the time precomputed on the host, they are simply stored in a look-up table. Possible
values are:
* 'od': Compute the time on-demand with the FPGA.
* 'lut': Compute the time on the host and store it in a lookup table.

StencilStream offers different backends or executors with different architectures or goals. The
possible values are:
* 'mono': Use the monotile FPGA backend of StencilStream. It yields a higher performance for the 
    same pipeline length than 'tiling', but it is limited to a maximal grid width and height.
* 'tiling': Use the tiling FPGA backend of StencilStream. It can handle arbitrarly large grids (and
    therefore cavity radii and resolutions), but generally yields a lower performance than 'mono'.
* 'cpu': Use the testing CPU backend of StencilStream. This backend is a trivial implementation of
    executor interface for CPUs and therefore performs worse than both FPGA backends when
    synthesized, but it's good enough to complete most simulations in reasonable times for testing
    purposes.

The FPGA backends also support some modifiers:
* 'emu': Don't synthesize the design and create an emulation image. Note that emulation is *very* 
    slow and not suitable to test the functionality of the transition function. Use the 'cpu'
    backend instead.
* 'report': Generate are hardware usage report for the variant. These reports can deliver a good 
    estimate on the synthesized design's performance.
EOF
exit 1
fi

MATERIAL=$1
SOURCE=$2
TIME=$3
BACKEND=$4
MODIFIER=$5

EXEC_NAME="fdtd_${MATERIAL}_${SOURCE}_${TIME}_${BACKEND}"
if [[ -n $MODIFIER ]]
then
    EXEC_NAME="${EXEC_NAME}_${MODIFIER}"
fi

COMMAND="dpcpp src/*.cpp -g -o $EXEC_NAME "
COMMAND="$COMMAND -std=c++20 -DSTENCIL_INDEX_WIDTH=32 -DFDTD_BURST_SIZE=1024 -qactypes -I./ -O3"

VALID_ARGUMENTS=1

# Noctua-specific options
if [[ -n "$EBROOTGCC" ]]
then
    COMMAND="$COMMAND --gcc-toolchain=$EBROOTGCC"
fi

# Material resolvers
if [[ "$MATERIAL" == "coef" ]]
then
    COMMAND="$COMMAND -DMATERIAL=0"
elif [[ "$MATERIAL" == "lut" ]]
then
    COMMAND="$COMMAND -DMATERIAL=1"
elif [[ "$MATERIAL" == "render" ]]
then
    COMMAND="$COMMAND -DMATERIAL=2"
else
    echo "Unknown material resolver '$MATERIAL'." 1>&2
    VALID_ARGUMENTS=0
fi

# Source wave computation
if [[ "$SOURCE" == "od" ]]
then
    COMMAND="$COMMAND -DSOURCE=0"
elif [[ "$SOURCE" == "lut" ]]
then
    COMMAND="$COMMAND -DSOURCE=1"
else
    echo "Unknown source type '$SOURCE'." 1>&2
    VALID_ARGUMENTS=0
fi

# Time computation
if [[ "$TIME" == "od" ]]
then
    COMMAND="$COMMAND -DTIME=0"
elif [[ "$TIME" == "lut" ]]
then
    COMMAND="$COMMAND -DTIME=1"
else
    echo "Unknown time type '$TIME'." 1>&2
    VALID_ARGUMENTS=0
fi

# FPGA-specific options
if [[ "$BACKEND" == "mono"  || "$BACKEND" == "tiling" ]]
then
    COMMAND="$COMMAND -fintelfpga -Xsv"

    if [[ "$MODIFIER" != "emu" ]]
    then
        COMMAND="$COMMAND -DHARDWARE -reuse-exe=$EXEC_NAME -Xshardware"
            
        if [[ -n $AOCL_BOARD_PACKAGE_ROOT ]]
        then
            COMMAND="$COMMAND -Xsboard=$FPGA_BOARD_NAME -Xsboard-package=$AOCL_BOARD_PACKAGE_ROOT"
        fi

        if [[ "$MODIFIER" == "report" ]]
        then
            COMMAND="$COMMAND -fsycl-link"
        fi
    fi
fi

if [[ "$BACKEND" == "mono" ]]
then
    COMMAND="$COMMAND -DEXECUTOR=0"
elif [[ "$BACKEND" == "tiling" ]]
then
    COMMAND="$COMMAND -DEXECUTOR=1"
elif [[ "$BACKEND" == "cpu" ]]
then
    COMMAND="$COMMAND -DEXECUTOR=2"
else
    echo "Unknown backend '$BACKEND'." 1>&2
    VALID_ARGUMENTS=0
fi

if [[ $VALID_ARGUMENTS == 0 ]]
then
    echo "Execute '$0 -h' to show the usage of this script." 1>&2
    exit 1
fi

echo $COMMAND
echo $COMMAND | bash
