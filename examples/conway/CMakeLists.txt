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
add_executable(conway_cpu conway.cpp)
target_compile_definitions(conway_cpu PUBLIC CPU)
add_dependencies(cpus conway_cpu)

# Common settings for all FPGA-based executables
add_compile_options(-fintelfpga)
add_link_options(-fintelfpga)

# The emulation image
add_executable(conway_emu conway.cpp)
target_compile_definitions(conway_emu PUBLIC EMULATOR)
add_dependencies(emus conway_emu)

# The hardware usage report
add_executable(conway_hw_report conway.cpp)
target_compile_definitions(conway_hw_report PUBLIC HARDWARE)
target_link_options(conway_hw_report PUBLIC -fintelfpga -Xshardware -fsycl-link)
add_dependencies(reports conway_hw_report)

# The hardware image
add_executable(conway_hw conway.cpp)
target_compile_definitions(conway_hw PUBLIC HARDWARE)
target_link_options(conway_hw PUBLIC -fintelfpga -Xshardware)
add_dependencies(hws conway_hw)

# Optionally adding the -Xsv option
if(VERBOSE_SYNTHESIS)
    target_compile_options(conway_hw_report conway_hw PUBLIC -Xsv)
endif()
