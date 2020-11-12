#!/bin/bash
#SBATCH -p fpgasyn
#SBATCH -o report.log
#SBATCH -J fdtd-report
#SBATCH --mail-type=ALL
#SBATCH --mem=8GB
#SBATCH --time=08:00:00

source /cm/shared/opt/intel_oneapi/beta-09/setvars.sh
module load nalla_pcie compiler/GCC lib/zlib devel/Boost

export HARDWARE=1
for i in 300
do
    export PIPELINE_LEN=$i
    make fdtd.report.tar.gz
    mv fdtd.report.tar.gz fdtd.$i.report.tar.gz
    rm -rf fdtd.prj *.a *.d
done
