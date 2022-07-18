#!/usr/bin/env bash

if [ $# -ne 4 ]
then
    echo "Usage: $0 <executable> <reference executable> <step size> <max generations>" 1>&2
    exit 1
fi

EXECUTABLE=$(realpath $1)
REFERENCE=$(realpath $2)
DATA=$(realpath data)
COMPARE_TEMPS=$(realpath ./scripts/compare_temps.sh)
STEP_SIZE=$3
MAX_GEN=$4

mkdir -p out
cd out

echo "ratio correct lines, average deviation, maximal deviation"

for i in $(seq $STEP_SIZE $STEP_SIZE $MAX_GEN)
do
    $EXECUTABLE 64 64 $i $DATA/temp_64 $DATA/power_64 out.$i.temp > /dev/null
    $REFERENCE 64 64 $i $(nproc) $DATA/temp_64 $DATA/power_64 ref.$i.temp > /dev/null
    $COMPARE_TEMPS out.$i.temp ref.$i.temp
done