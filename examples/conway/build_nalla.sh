#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J conway-build
#SBATCH --mail-type=ALL
#SBATCH --mem=90GB
#SBATCH --time=3-00:00:00

../build.sh nalla_pcie/19.4.0_max 10 conway