#!/usr/bin/env bash

module reset
module load compiler Clang
clang-format -i --style='{ BasedOnStyle: LLVM, IndentWidth: 4, TabWidth: 4, ColumnLimit: 100, IndentPPDirectives: BeforeHash }' $(find -path '*.*pp' ! -path './build/*' ! -path './examples/hotspot/hotspot_openmp.cpp' ! -path './examples/hotspot/data/*')
