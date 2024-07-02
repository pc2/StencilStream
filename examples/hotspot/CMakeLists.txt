# Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Reference benchmark
add_executable(hotspot_openmp hotspot_openmp.cpp)
target_compile_options(hotspot_openmp PUBLIC -g -fopenmp -O2)
target_link_options(hotspot_openmp PUBLIC -g -fopenmp -O2)

# Global settings
add_compile_definitions(STENCIL_INDEX_WIDTH=32)

# Pure CPU-based executable
add_executable(hotspot_cpu hotspot.cpp)
target_link_libraries(hotspot_cpu PUBLIC StencilStream_CPU)

# Monotile versions
add_executable(hotspot_mono hotspot.cpp)
target_link_libraries(hotspot_mono PUBLIC StencilStream_Monotile)

add_executable(hotspot_mono_emu hotspot.cpp)
target_link_libraries(hotspot_mono_emu PUBLIC StencilStream_MonotileEmulator)

add_executable(hotspot_mono_report hotspot.cpp)
target_link_libraries(hotspot_mono_report PUBLIC StencilStream_MonotileReport)

# Tiling versions
add_executable(hotspot_tiling hotspot.cpp)
target_link_libraries(hotspot_tiling PUBLIC StencilStream_Tiling)

add_executable(hotspot_tiling_emu hotspot.cpp)
target_link_libraries(hotspot_tiling_emu PUBLIC StencilStream_TilingEmulator)

add_executable(hotspot_tiling_report hotspot.cpp)
target_link_libraries(hotspot_tiling_report PUBLIC StencilStream_TilingReport)

# Limiting clock speed and setting the seed for Hotspot Tiling
foreach(EXECUTABLE hotspot_tiling_report hotspot_tiling)
    target_link_options(${EXECUTABLE} PUBLIC -Xsclock=300MHz -Xsseed=3)
endforeach()

# Adding a custom target for all Hotspot reports
add_custom_target(hotspot_reports)
add_dependencies(hotspot_reports hotspot_mono_report hotspot_tiling_report)
