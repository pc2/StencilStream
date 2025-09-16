#!/usr/bin/env bash
#SBATCH --account=hpc-lco-kenter   # Use your project's account name
#SBATCH --gres=gpu:a100:1
##SBATCH --qos=devel --partition=dgx
#SBATCH -c 32 --mem 32G -t 00:05:00
#SBATCH --output=%j_hotspot_cuda.out


ml reset
ml devel/CMake/3.29.3-GCCcore-13.3.0
ml fpga
ml lang
ml intel/oneapi/25.0.0
ml system/CUDA/12.6.0

cd /scratch/hpc-lco-kenter/tstoehr/stencilstream-prototype/build
echo "Starting program"

cd /scratch/hpc-lco-kenter/tstoehr/sycl-stencil/build_new

for ((i=0; i<5; i++)); do
    ./examples/hotspot/hotspot_cuda 1024 1024 1000 temp_1024 power_1024 out.txt
done






