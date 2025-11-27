#!/usr/bin/env bash
#SBATCH -A pc2-mitarbeiter -p normal -q fpgasynthesis -t 1-00:00:00 -c 8 --mem 120G
#SBATCH --array=0-7
#SBATCH --mail-type=ALL --mail-user=joo@mail.upb.de

task_id=${SLURM_ARRAY_TASK_ID}
targets=("Jacobi1General" "Jacobi2Constant" "Jacobi3Constant" "Jacobi4Constant" "Jacobi5Constant" "Jacobi4General" "Jacobi9General")

source scripts/env_fpga.sh
cd build
CHANNEL_MAPPING_MODE=2 make "${targets[$task_id]}_mono"
