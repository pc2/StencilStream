#!/usr/bin/env bash
#SBATCH -A pc2-mitarbeiter -p gpu --gres=gpu:a100:1 -c 16 --mem 128G -t 02:00:00
#SBATCH --mail-type=ALL --mail-user=joo@mail.upb.de -J jacobi_cuda

./scripts/benchmark.jl deep_grid_scaling ../../build/examples/jacobi/Jacobi5General_cuda 1