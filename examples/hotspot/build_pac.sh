#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J hotspot-build
#SBATCH --mail-type=ALL
#SBATCH --exclusive
#SBATCH --time=3-00:00:00

../build.sh intel_pac/19.2.0 200 hotspot