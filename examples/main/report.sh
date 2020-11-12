#!/bin/bash
#SBATCH -A hpc-lco-kenter
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J main-build
#SBATCH --mail-type=ALL
#SBATCH --mem=8GB
#SBATCH --time=00:15:00

source /cm/shared/opt/intel_oneapi/beta-09/setvars.sh
module load nalla_pcie compiler/GCC 

export HARDWARE=1
export PIPELINE_LEN=10

time make main.report.tar.gz
