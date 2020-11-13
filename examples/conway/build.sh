#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J conway-build
#SBATCH --mail-type=ALL
#SBATCH --mem=75GB
#SBATCH --time=3-00:00:00

source /cm/shared/opt/intel_oneapi/beta-10/setvars.sh
module load nalla_pcie compiler/GCC 

export HARDWARE=1
export PIPELINE_LEN=10

time make conway
tar -cf - conway conway.prj/reports | ~/pigz > lean.tar.gz &
tar -cf - conway conway.prj | ~/pigz > full.tar.gz &
wait
rm -r conway.prj