#!/usr/bin/env bash
module reset
#ml lang fpga devel system intel/oneapi/25.0.0 CMake/3.29.3-GCCcore-13.3.0 CUDA/12.6.0

ml fpga
ml intel/oneapi/25.0.0
ml devel/CMake/3.29.3-GCCcore-13.3.0
ml system/CUDA/12.6.0

icpx --version
nvidia-smi