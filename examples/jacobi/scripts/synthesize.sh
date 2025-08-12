#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter -p normal -q fpgasynthesis --array=0-15 -c 8 --mem 120G -t 1-00:00:00

task_id=${SLURM_ARRAY_TASK_ID}
variant=$1
targets=("Jacobi1General" "Jacobi2Constant" "Jacobi3Constant" "Jacobi4Constant" "Jacobi5Constant" "Jacobi4General" "Jacobi5General" "Jacobi9General")
make "${targets[$task_id]}_$variant"
