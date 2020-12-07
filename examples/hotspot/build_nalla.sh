#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J hotspot-build
#SBATCH --mail-type=ALL
#SBATCH --exclusive
#SBATCH --time=3-00:00:00

../build.sh nalla_pcie/19.4.0_max 200 hotspot