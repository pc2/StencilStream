#!/usr/bin/env bash
module reset

ml system/ROCM/6.3.3
ml fpga
ml lang
ml intel/oneapi/25.0.0
ml system/CUDA/12.6.0
ml fpga/xilinx/xrt/2.16
ml devel/CMake/3.29.3-GCCcore-13.3.0