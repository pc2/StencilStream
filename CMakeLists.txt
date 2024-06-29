# Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(MAIN_PROJECT ON)
    set(CMAKE_CXX_COMPILER icpx)
else()
    set(MAIN_PROJECT OFF)
endif()

cmake_minimum_required(VERSION 3.21)

project(StencilStream CXX)
set(CMAKE_PROJECT_VERSION 4.0.0)

if(${MAIN_PROJECT})
    add_subdirectory(examples/convection)
    add_subdirectory(examples/conway)
    add_subdirectory(examples/fdtd)
    add_subdirectory(examples/hotspot)
    add_subdirectory(tests)
endif()

find_package(Boost 1.74.0 REQUIRED)

add_library(StencilStream INTERFACE)
target_include_directories(StencilStream INTERFACE "${PROJECT_SOURCE_DIR}/")
target_compile_options(StencilStream INTERFACE -fsycl -std=c++20)
target_link_options(StencilStream INTERFACE -fsycl)
target_link_libraries(StencilStream INTERFACE Boost::boost)

add_library(StencilStreamCPU INTERFACE)
target_link_libraries(StencilStreamCPU INTERFACE StencilStream)

add_library(StencilStreamFPGA INTERFACE)
target_link_libraries(StencilStreamFPGA INTERFACE StencilStream)
target_compile_options(StencilStreamFPGA INTERFACE -qactypes -fintelfpga)
target_link_options(StencilStreamFPGA INTERFACE -fintelfpga -Xshardware)

add_library(StencilStreamFPGAEmulator INTERFACE)
target_link_libraries(StencilStreamFPGAEmulator INTERFACE StencilStream)
target_compile_options(StencilStreamFPGAEmulator INTERFACE -qactypes -fintelfpga)
target_link_options(StencilStreamFPGAEmulator INTERFACE -fintelfpga)

add_library(StencilStreamFPGAReport INTERFACE)
target_link_libraries(StencilStreamFPGAReport INTERFACE StencilStream)
target_compile_options(StencilStreamFPGAReport INTERFACE -qactypes -fintelfpga)
target_link_options(StencilStreamFPGAReport INTERFACE -fintelfpga -Xshardware -fsycl-link)
