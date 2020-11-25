#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o build.log
#SBATCH -J conway-build
#SBATCH --mail-type=ALL
#SBATCH --mem=90GB
#SBATCH --time=3-00:00:00

../build.sh intel_pac/19.2.0_usm 10 conway