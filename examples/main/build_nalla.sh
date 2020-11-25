#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J main-build
#SBATCH --mail-type=ALL
#SBATCH --mem=60GB
#SBATCH --time=3-00:00:00

../build.sh nalla_pcie 10 main