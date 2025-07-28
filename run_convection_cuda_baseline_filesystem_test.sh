#!/usr/bin/env bash
#SBATCH --account=hpc-lco-kenter   # Use your project's account name

#SBATCH --gres=gpu:a100:1
##SBATCH --qos=devel --partition=dgx
#SBATCH -c 32 --mem 32G -t 00:15:00
#SBATCH --output=%j_convection_baseline_filesystem_test.out

ml reset
ml devel/CMake/3.29.3-GCCcore-13.3.0
ml fpga
ml lang
ml intel/oneapi/25.0.0
ml system/CUDA/12.6.0


sbatch ../run_convection_cuda_baseline_max_res.sh
sbatch ../run_convection_cuda_baseline_max_res_dgx.sh