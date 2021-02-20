#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J main-build
#SBATCH --mail-type=ALL
#SBATCH --mem=60GB
#SBATCH --time=3-00:00:00

source /cm/shared/opt/intel_oneapi/beta-10/setvars.sh
module load $BSP_MODULE compiler/GCC

make execution.report.tar.gz