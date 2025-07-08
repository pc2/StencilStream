#!/usr/bin/env bash
module reset
ml lang fpga devel intel/oneapi/24.2.1 CMake/3.29.3-GCCcore-13.3.0 JuliaHPC/1.11.4-foss-2024a-CUDA-12.6.0
icpx --version
julia --version
echo "julia depot path: $JULIA_DEPOT_PATH"

if [[ ! -e Manifest.toml ]];
then
    julia --project -e "using Pkg; Pkg.instantiate()"
fi