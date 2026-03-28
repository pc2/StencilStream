#!/usr/bin/env bash
#SBATCH -A pc2-mitarbeiter -p gpu --gres=gpu:a100:1 -c 16 --mem 128G -t 02:00:00
#SBATCH --mail-type=ALL --mail-user=joo@mail.upb.de -J fdtd_cuda

./scripts/benchmark.jl deep_grid_scaling ../../build/examples/fdtd/fdtd_coef_device_cuda cuda 1
./scripts/benchmark.jl deep_grid_scaling_ncu ../../build/examples/fdtd/fdtd_coef_device_cuda cuda 1
./scripts/benchmark.jl deep_grid_scaling_ncu ../../build/examples/fdtd/fdtd_coef_device_cuda_naive cuda_naive 1