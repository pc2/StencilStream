#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter -p normal -q fpgasynthesis --array=0-15 -c 8 --mem 120G -t 1-00:00:00

task_id=${SLURM_ARRAY_TASK_ID}
targets=("Jacobi1General_mono" "Jacobi1General_tiling" "Jacobi2Constant_mono" "Jacobi2Constant_tiling" "Jacobi3Constant_mono" "Jacobi3Constant_tiling" "Jacobi4Constant_mono" "Jacobi4Constant_tiling" "Jacobi5Constant_mono" "Jacobi5Constant_tiling" "Jacobi4General_mono" "Jacobi4General_tiling" "Jacobi5General_mono" "Jacobi5General_tiling" "Jacobi9General_mono" "Jacobi9General_tiling")
make ${targets[$task_id]}
