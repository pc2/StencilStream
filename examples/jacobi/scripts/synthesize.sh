#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter -p normal -q fpgasynthesis --array=0-4 -c 8 --mem 120G -t 1-00:00:00

task_id=${SLURM_ARRAY_TASK_ID}
variant=$1
#targets=("Jacobi1General" "Jacobi2Constant" "Jacobi3Constant" "Jacobi4Constant" "Jacobi5Constant" "Jacobi4General" "Jacobi5General" "Jacobi9General")
targets=("Jacobi1General" "Jacobi2Constant" "Jacobi3Constant" "Jacobi4Constant" "Jacobi9General")

mkdir -p build
cd build

if [ $task_id == 0 ];
then
    cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-O0 ..
else
    until [ -f Makefile ]
    do
        sleep 60
    done
fi

make "${targets[$task_id]}_$variant"
