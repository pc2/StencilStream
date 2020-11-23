#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J conway-build
#SBATCH --mail-type=ALL
#SBATCH --mem=75GB
#SBATCH --time=3-00:00:00

source /cm/shared/opt/intel_oneapi/beta-10/setvars.sh
module load compiler/GCC 

echo "Building for Board $FPGA_BOARD_NAME"

function archive_build {
    tar -cf - conway conway.prj/reports | ~/pigz > lean.tar.gz &
    tar -cf - conway conway.prj | ~/pigz > full.tar.gz &
    wait
    rm -r conway.prj
}

export HARDWARE=1
export PIPELINE_LEN=10

time make conway && archive_build