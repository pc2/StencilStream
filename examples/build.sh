#!/bin/bash

export HARDWARE=1
export BSP_MODULE=$1
export PIPELINE_LEN=$2
export BIN_NAME=$3

source /cm/shared/opt/intel_oneapi/beta-10/setvars.sh
module load $BSP_MODULE compiler/GCC lib/zlib devel/Boost

echo "Building for Board $FPGA_BOARD_NAME"

function archive_build {
    tar -cf - $BIN_NAME $BIN_NAME.prj/reports | ~/pigz > lean.$FPGA_BOARD_NAME.tar.gz &
    tar -cf - $BIN_NAME $BIN_NAME.prj | ~/pigz > full.$FPGA_BOARD_NAME.tar.gz &
    wait
    rm -r $BIN_NAME.prj
}

time make $BIN_NAME && archive_build
