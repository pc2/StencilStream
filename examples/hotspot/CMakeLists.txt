# Copyright © 2020-2022Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Global settings
include_directories(.)
add_compile_options(-std=c++20 -qactypes)
add_compile_definitions(STENCIL_INDEX_WIDTH=32)

# Pure CPU-based executable
add_executable(hotspot_cpu hotspot.cpp)
target_compile_definitions(hotspot_cpu PUBLIC EXECUTOR=2)

# Common hardware options
add_compile_options(-fintelfpga)
add_link_options(-fintelfpga -Xshardware)

if(VERBOSE_SYNTHESIS)
    add_link_options(-Xsv)
endif()

# Monotile versions
foreach(EXECUTABLE hotspot_mono hotspot_mono_report)
    add_executable(${EXECUTABLE} hotspot.cpp)
    target_compile_definitions(${EXECUTABLE} PUBLIC EXECUTOR=0)
endforeach()

# Tiling versions
foreach(EXECUTABLE hotspot_tiling hotspot_tiling_report)
    add_executable(${EXECUTABLE} hotspot.cpp)
    target_compile_definitions(${EXECUTABLE} PUBLIC EXECUTOR=1)
endforeach()

# Adding -fsycl-link to report executables
foreach(EXECUTABLE hotspot_mono_report hotspot_tiling_report)
    target_link_options(${EXECUTABLE} PUBLIC -fsycl-link)
endforeach()

add_dependencies(cpus hotspot_cpu)
add_dependencies(reports hotspot_mono_report hotspot_tiling_report)
add_dependencies(hws hotspot_mono hotspot_tiling)

