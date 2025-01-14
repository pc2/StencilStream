#!/usr/bin/env bash
module reset
ml lang fpga devel intel/oneapi/25.0.0 bittware/520n/20.4.0_hpc CMake/3.22.1-GCCcore-11.2.0 Julia/1.10.4
icpx --version
julia --version
echo "julia depot path: $JULIA_DEPOT_PATH"

if [[ ! -e Manifest.toml ]];
then
    julia --project -e "using Pkg; Pkg.instantiate()"
fi