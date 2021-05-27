#!/usr/bin/env bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J StencilStream-synthesis-test
#SBATCH --mail-type=ALL
#SBATCH --mem=90000MB
#SBATCH --time=24:00:00

source /cm/shared/opt/intel_oneapi/2021.1.1/setvars.sh
module load compiler/GCC nalla_pcie
make synthesis_hw