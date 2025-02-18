#!/usr/bin/env bash
module reset

ml fpga
ml lang
ml intel/oneapi/24.2.1
ml CMake/3.29.3-GCCcore-13.3.0 Julia/1.10.4
ml system/CUDA/12.6.0

echo "Julia depot path: $JULIA_DEPOT_PATH"

if [[ ! -e Manifest.toml ]];
then
    julia --project -e "using Pkg; Pkg.instantiate()"
fi

nvidia-smi
icpx --version
julia --version
