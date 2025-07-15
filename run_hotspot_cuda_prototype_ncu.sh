#!/usr/bin/env bash
#SBATCH --account=hpc-lco-kenter   # Use your project's account name
#SBATCH --gres=gpu:a100:1

#SBATCH -c 32 --mem 32G -t 00:05:00
#SBATCH --output=%j_ncu_hotspot_prototype.out


ml reset
ml devel/CMake/3.29.3-GCCcore-13.3.0
ml fpga
ml lang
ml intel/oneapi/25.0.0
ml system/CUDA/12.6.0

cd /scratch/hpc-lco-kenter/tstoehr/stencilstream-prototype/build
echo "Starting program"

ncu --set full -o hotspot_cuda_prototype ./src/hotspot/hotspot_cuda_prototype 1024 1024 1000 temp_1024 power_1024 out.txt