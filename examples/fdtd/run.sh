#!/bin/bash
#SBATCH -p fpga
#SBATCH -o benchmark.log
#SBATCH -J fdtd-benchmark
#SBATCH --constraint=19.4.0_max
#SBATCH --mail-type=ALL
#SBATCH --time=30:00

source /cm/shared/opt/intel_oneapi/beta-10/setvars.sh
module load nalla_pcie compiler/GCC lib/zlib devel/Boost

mkdir -p output
cd output
../fdtd -t 60000 -c 60
