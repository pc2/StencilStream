#!/usr/bin/env bash

source /cm/shared/opt/intel_oneapi/2021.3/setvars.sh
ml compiler/GCC nalla_pcie/20.4.0

make all -j$(nprocs)