#!/usr/bin/env bash
#SBATCH -J hotspot-synthesis
#SBATCH -o synthesize.log
#SBATCH -p fpgasyn
#SBATCH --mail-type=all
#SBATCH --mem=120000MB
#SBATCH --time=48:00:00

source /cm/shared/opt/intel_oneapi/2021.2/setvars.sh
ml compiler/GCC nalla_pcie/19.4.0

make hotspot_hw