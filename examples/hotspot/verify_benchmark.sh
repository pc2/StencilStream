#!/bin/bash
#SBATCH -p short
#SBATCH -o verify_benchmark.log
#SBATCH -J verify-hotspot-benchmark
#SBATCH --mail-type=ALL

module load compiler/GCC

export RODINIA=$PC2DATA/hpc-prf-hpmesh/joo/rodinia/openmp/hotspot/hotspot
export STEP_SIZE=250
export RUNS=`seq 1 100`

echo "value_derivation = {"

export OUT_DIR=output.$STEP_SIZE
echo "    $STEP_SIZE: ["
for i in $RUNS
do
    $RODINIA 1024 1024 $(($i*$STEP_SIZE)) 40 data/temp_1024 data/power_1024 $OUT_DIR/ref.$i 2>/dev/null >/dev/null

    paste $OUT_DIR/out.$i $OUT_DIR/ref.$i | tr " \t" "," | cut -d, -f2,4 | awk -F "," \
        'BEGIN {n_correct = 0; n_lines = 0; diff_sum = 0; diff_max = -1; }

        {
            n_lines++;
            if ($1 == $2) {
                n_correct++;
            }
            diff = $1 - $2;
            if (diff < 0) {
                diff *= -1;
            }
            if (diff_max < diff) {
                diff_max = diff;
            }
            diff_sum += diff;
        }

        END {print "        (" n_correct/n_lines ", " diff_sum/n_lines ", " diff_max "),"}'
    done
echo "    ],"

echo "}"
