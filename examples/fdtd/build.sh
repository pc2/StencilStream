#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J fdtd-build
#SBATCH --mail-type=ALL
#SBATCH --exclusive
#SBATCH --time=3-00:00:00

source /cm/shared/opt/intel_oneapi/beta-10/setvars.sh
module load compiler/GCC lib/zlib devel/Boost

echo "Building for Board $FPGA_BOARD_NAME"

function archive_build {
    tar -cf - fdtd fdtd.prj/reports | ~/pigz > lean.$FPGA_BOARD_NAME.tar.gz &
    tar -cf - fdtd fdtd.prj | ~/pigz > full.$FPGA_BOARD_NAME.tar.gz &
    wait
    rm -r fdtd.prj
}

export HARDWARE=1
export PIPELINE_LEN=40

time make fdtd && archive_build
