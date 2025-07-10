#!/usr/bin/env bash
#SBATCH --account=hpc-lco-kenter   # Use your project's account name
##SBATCH --gres=gpu:a100_1g.5gb:1
##SBATCH --qos=devel --partition=dgx

#SBATCH --gres=gpu:a100:1
#SBATCH -c 32 --mem 32G -t 00:01:00
#SBATCH --output=%j_convection_baseline_nsys.out

ml reset
ml devel/CMake/3.29.3-GCCcore-13.3.0
ml fpga
ml lang
ml intel/oneapi/25.0.0
ml system/CUDA/12.6.0

cd /scratch/hpc-lco-kenter/tstoehr/stencilstream-prototype/build
nsys profile ./scratch/hpc-lco-kenter/tstoehr/sycl-stencil/build/examples/convection/convection_cuda_baseline /scratch/hpc-lco-kenter/tstoehr/sycl-stencil/examples/convection/experiments/max-res-default.json /scratch/hpc-lco-kenter/tstoehr/sycl-stencil/build/out
