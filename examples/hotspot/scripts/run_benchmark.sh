#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter -p fpga --constraint=bittware_520n_20.4.0_hpc
#SBATCH -c 16 --mem 128G -t 00:30:00
#SBATCH --mail-type=ALL --mail-user=joo@mail.upb.de -J hotspot
./scripts/benchmark.jl max_perf ../../build/examples/hotspot/hotspot_tiling tiling