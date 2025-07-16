#!/usr/bin/env bash
#SBATCH -A hpc-lco-kenter -p fpga --constraint=bittware_520n_20.4.0_max
#SBATCH --mem 128G -t 04:00:00
#SBATCH --mail-type=ALL --mail-user=joo@mail.upb.de -J fdtd

../../build/examples/fdtd/fdtd_coef_device_tiling -c experiments/max_grid.json -o out/