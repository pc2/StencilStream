#!/usr/bin/env bash
module reset
module load lang fpga devel intel/oneapi/23.2.0 bittware/520n/20.4.0_hpc Boost/1.81.0-GCC-12.2.0 CMake/3.24.3-GCCcore-12.2.0 Julia/1.10.0-linux-x86_64
icpx --version
julia --version
echo "julia depot path: $JULIA_DEPOT_PATH"

if [[ ! -e Manifest.toml ]];
then
    julia --project -e "using Pkg; Pkg.instantiate()"
fi