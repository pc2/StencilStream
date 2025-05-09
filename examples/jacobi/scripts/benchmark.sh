#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter -p fpga --constraint=bittware_520n_20.4.0_hpc -t 03:00:00

for exe in ../../build/examples/jacobi/Jacobi*_mono ../../build/examples/jacobi/Jacobi*_tiling
do
    ./scripts/benchmark.jl max_perf $exe
done