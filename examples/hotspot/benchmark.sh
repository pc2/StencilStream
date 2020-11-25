#!/bin/bash
#SBATCH -p fpga
#SBATCH -o benchmark.py
#SBATCH -J hotspot-benchmark
#SBATCH --constraint=19.4.0_max
#SBATCH --mail-type=ALL
#SBATCH --time=30:00

export BSP_MODULE=$1

source /cm/shared/opt/intel_oneapi/beta-10/setvars.sh
module load $BSP_MODULE compiler/GCC 

export STEP_SIZE=225
export RUNS=`seq 1 100`

echo "runtime = {"
echo "    $STEP_SIZE: ["

export OUT_DIR=output.$STEP_SIZE
rm -rf $OUT_DIR
mkdir $OUT_DIR

for i in $RUNS
do
    ./hotspot 1024 1024 $(($i*$STEP_SIZE)) data/temp_1024 data/power_1024 $OUT_DIR/out.$i | \
        tr " " "\n" | \
        awk '/[0-9]+\.[0-9]+/ {print "        " $0 ","}'
done

echo "   ],"
echo "}"
