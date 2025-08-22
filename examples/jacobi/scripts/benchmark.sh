#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter -p fpga --constraint=bittware_520n_20.4.0_max -t 01:00:00
#SBATCH -n 24 --tasks-per-node=2 -x "n2fpga[20,22,23,26]"
#SBATCH --mail-user=joo@mail.upb.de --mail-type=ALL

./scripts/benchmark.jl max_perf ../../build/examples/jacobi/ 24