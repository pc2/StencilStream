#!/usr/bin/env bash
#SBATCH -A pc2-mitarbeiter -p fpga --constraint=bittware_520n_20.4.0_max
#SBATCH -n 16 --ntasks-per-node=2 -x "n2fpga[20,22,23,26]" -t 02:00:00
#SBATCH --mail-type=ALL --mail-user=joo@mail.upb.de -J jacobi_mono

./scripts/benchmark.jl deep_grid_scaling ../../build/Jacobi5General_mono 16