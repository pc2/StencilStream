#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J main-build
#SBATCH --mail-type=ALL
#SBATCH --exclusive
#SBATCH --time=3-00:00:00

source /cm/shared/opt/intel_oneapi/beta-10/setvars.sh
module load compiler/GCC lib/zlib devel/Boost

echo "Building for Board $FPGA_BOARD_NAME"

function archive_build {
    tar -cf - main main.prj/reports | ~/pigz > lean.$FPGA_BOARD_NAME.tar.gz &
    tar -cf - main main.prj | ~/pigz > full.$FPGA_BOARD_NAME.tar.gz &
    wait
    rm -r main.prj
}

export HARDWARE=1
export PIPELINE_LEN=10

time make main && archive_build
