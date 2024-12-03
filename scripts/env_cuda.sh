#!/usr/bin/env bash
module reset
ml fpga devel compiler
ml intel/oneapi-llvm
ml system/CUDA/12.6.0
ml devel/Boost/1.83.0-GCC-13.2.0
ml devel/CMake/3.29.3-GCCcore-13.3.0
icpx --version