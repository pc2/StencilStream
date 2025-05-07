#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter -p normal -q fpgasynthesis --array=0-7 -c 8 --mem 120G -t 1-00:00:00

task_id=${SLURM_ARRAY_TASK_ID}
targets=("Jacobi5Constant_mono" "Jacobi5Constant_tiling" "Jacobi4General_mono" "Jacobi4General_tiling" "Jacobi5General_mono" "Jacobi5General_tiling" "Jacobi9General_mono" "Jacobi9General_tiling")
make ${targets[$task_id]}
