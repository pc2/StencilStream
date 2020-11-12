#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o report.log
#SBATCH -J hotspot-report
#SBATCH --mail-type=ALL
#SBATCH --mem=8GB
#SBATCH --time=08:00:00

source /cm/shared/opt/intel_oneapi/beta-10/setvars.sh
module load nalla_pcie compiler/GCC 

for i in 250
do
    export PIPELINE_LEN=$i
    make hotspot.report.tar.gz
    mv hotspot.report.tar.gz hotspot.$i.report.tar.gz
    rm -rf hotspot.prj *.a *.d
done
