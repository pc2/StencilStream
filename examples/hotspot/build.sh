#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J hotspot-build
#SBATCH --mail-type=ALL
#SBATCH --exclusive
#SBATCH --time=3-00:00:00

source /cm/shared/opt/intel_oneapi/beta-10/setvars.sh
module load nalla_pcie compiler/GCC 

function archive_build {
    tar -cf - hotspot hotspot.prj/reports | ~/pigz > lean.tar.gz &
    tar -cf - hotspot hotspot.prj | ~/pigz > full.tar.gz &
    wait
    rm -r hotspot.prj
}

export HARDWARE=1
export PIPELINE_LEN=250

time make hotspot && archive_build