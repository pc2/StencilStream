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

option(StencilStream_VerboseSynthesis "Print some progress information during hardware synthesis" OFF)

if(${MAIN_PROJECT})
    add_subdirectory(examples/convection)
    add_subdirectory(examples/conway)
    add_subdirectory(examples/fdtd)
    add_subdirectory(examples/hotspot)
    add_subdirectory(tests)
endif()

find_package(Boost 1.74.0 REQUIRED)

# Base target

add_library(StencilStream_Base INTERFACE)
target_include_directories(StencilStream_Base INTERFACE "${PROJECT_SOURCE_DIR}/")
target_compile_options(StencilStream_Base INTERFACE -fsycl -std=c++20)
target_link_options(StencilStream_Base INTERFACE -fsycl)
target_link_libraries(StencilStream_Base INTERFACE Boost::boost)

# Target for CPU backend

add_library(StencilStream_CPU INTERFACE)
target_link_libraries(StencilStream_CPU INTERFACE StencilStream_Base)
target_compile_definitions(StencilStream_CPU INTERFACE STENCILSTREAM_BACKEND_CPU=1)
target_compile_definitions(StencilStream_CPU INTERFACE STENCILSTREAM_TARGET_CPU=1)

# Base target for FPGA-based backends

add_library(StencilStream_FPGABase INTERFACE)
target_link_libraries(StencilStream_FPGABase INTERFACE StencilStream_Base)
target_compile_options(StencilStream_FPGABase INTERFACE -qactypes -fintelfpga)
target_link_options(StencilStream_FPGABase INTERFACE -fintelfpga)

if(StencilStream_VerboseSynthesis)
    target_link_options(StencilStream_FPGABase INTERFACE -Xsv)
endif() 

# Monotile FPGA backends

add_library(StencilStream_MonotileBase INTERFACE)
target_link_libraries(StencilStream_MonotileBase INTERFACE StencilStream_FPGABase)
target_compile_definitions(StencilStream_MonotileBase INTERFACE STENCILSTREAM_BACKEND_MONOTILE=1)

add_library(StencilStream_MonotileEmulator INTERFACE)
target_link_libraries(StencilStream_MonotileEmulator INTERFACE StencilStream_MonotileBase)
target_compile_definitions(StencilStream_MonotileEmulator INTERFACE STENCILSTREAM_TARGET_FPGA_EMU=1)

add_library(StencilStream_Monotile INTERFACE)
target_link_libraries(StencilStream_Monotile INTERFACE StencilStream_MonotileBase)
target_compile_definitions(StencilStream_Monotile INTERFACE STENCILSTREAM_TARGET_FPGA=1)
target_link_options(StencilStream_Monotile INTERFACE -Xshardware)

add_library(StencilStream_MonotileReport INTERFACE)
target_link_libraries(StencilStream_MonotileReport INTERFACE StencilStream_MonotileBase)
target_compile_definitions(StencilStream_MonotileReport INTERFACE STENCILSTREAM_TARGET_FPGA=1)
target_link_options(StencilStream_MonotileReport INTERFACE -Xshardware -fsycl-link)

# Tiling FPGA backends

add_library(StencilStream_TilingBase INTERFACE)
target_link_libraries(StencilStream_TilingBase INTERFACE StencilStream_FPGABase)
target_compile_definitions(StencilStream_TilingBase INTERFACE STENCILSTREAM_BACKEND_TILING=1)

add_library(StencilStream_TilingEmulator INTERFACE)
target_link_libraries(StencilStream_TilingEmulator INTERFACE StencilStream_TilingBase)
target_compile_definitions(StencilStream_TilingEmulator INTERFACE STENCILSTREAM_TARGET_FPGA_EMU=1)

add_library(StencilStream_Tiling INTERFACE)
target_link_libraries(StencilStream_Tiling INTERFACE StencilStream_TilingBase)
target_compile_definitions(StencilStream_Tiling INTERFACE STENCILSTREAM_TARGET_FPGA=1)
target_link_options(StencilStream_Tiling INTERFACE -Xshardware)

add_library(StencilStream_TilingReport INTERFACE)
target_link_libraries(StencilStream_TilingReport INTERFACE StencilStream_TilingBase)
target_compile_definitions(StencilStream_TilingReport INTERFACE STENCILSTREAM_TARGET_FPGA=1)
target_link_options(StencilStream_TilingReport INTERFACE -Xshardware -fsycl-link)
