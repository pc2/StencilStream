#!/usr/bin/env bash
#SBATCH -A pc2-mitarbeiter -p gpu --gres=gpu:a100:1 -c 16 --mem 128G -t 01:00:00
#SBATCH --mail-type=ALL --mail-user=joo@mail.upb.de -J hotspot_cuda

./scripts/benchmark.jl deep_grid_scaling ../../build/examples/hotspot/hotspot_cuda cuda 1