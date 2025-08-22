#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter -p fpga --constraint=bittware_520n_20.4.0_max
#SBATCH -n 24 --ntasks-per-node=2 -x "n2fpga[20,22,23,26]" -t 01:00:00
#SBATCH --mail-type=ALL --mail-user=joo@mail.upb.de -J fdtd

./scripts/benchmark.jl strong_scaling ../../build/examples/fdtd/fdtd_coef_device_mono monotile 24