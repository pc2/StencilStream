#!/usr/bin/env bash
#SBATCH -J fdtd-synthesis
#SBATCH -o synthesize.log
#SBATCH -p fpgasyn
#SBATCH --mail-type=all
#SBATCH --exclusive
#SBATCH --time=24:00:00

source /cm/shared/opt/intel_oneapi/2021.1.1/setvars.sh
ml compiler/GCC nalla_pcie/19.4.0

make fdtd_hw