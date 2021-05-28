#!/usr/bin/env bash
#SBATCH -J fdtd-simulate
#SBATCH -p fpga
#SBATCH --time=00:05:00
#SBATCH -o simulate.log
#SBATCH --constraint=19.4.0_max
#SBATCH --mail-type=all

source /cm/shared/opt/intel_oneapi/2021.1.1/setvars.sh
ml compiler/GCC nalla_pcie

mkdir -p output
cd output
../fdtd_hw -c 1024 -f 300