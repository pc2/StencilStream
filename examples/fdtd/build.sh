#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J fdtd-build
#SBATCH --mail-type=ALL
#SBATCH --exclusive
#SBATCH --time=3-00:00:00

source /cm/shared/opt/intel_oneapi/beta-10/setvars.sh
module load nalla_pcie compiler/GCC lib/zlib devel/Boost

function archive_build {
    tar -cf - fdtd fdtd.prj/reports | ~/pigz > lean.tar.gz &
    tar -cf - fdtd fdtd.prj | ~/pigz > full.tar.gz &
    wait
    rm -r fdtd.prj
}

export HARDWARE=1
export PIPELINE_LEN=30

time make fdtd && archive_build
