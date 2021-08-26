#!/usr/bin/env bash

source /cm/shared/opt/intel_oneapi/2021.1.1/setvars.sh
ml compiler/GCC nalla_pcie/19.4.0

make all -j$(nprocs)