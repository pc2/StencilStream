# Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

set(UNIT_TEST_BASE_SOURCES
    main.cpp
    HostPipe.cpp
    Stencil.cpp
)

set(UNIT_TEST_CPU_SOURCES
    cpu/Grid.cpp
    cpu/StencilUpdate.cpp
)

set(UNIT_TEST_CUDA_SOURCES
    cuda/Grid.cpp
    cuda/StencilUpdate.cpp
)

set(UNIT_TEST_FPGA_SOURCES
    fpga_io/DualIOPipeKernels.cpp
    fpga_io/MemoryKernels.cpp
    monotile/Grid.cpp
    monotile/StencilUpdate.cpp
    monotile/StencilUpdateKernel.cpp
    tiling/Grid.cpp
    tiling/HaloInjectionKernel.cpp
    tiling/StencilUpdate.cpp
    tiling/StencilUpdateKernel.cpp
)

add_executable(unit_test_cpu ${UNIT_TEST_BASE_SOURCES} ${UNIT_TEST_CPU_SOURCES})
target_link_libraries(unit_test_cpu PRIVATE Catch2::Catch2 StencilStream_CPU)

add_executable(unit_test_cuda ${UNIT_TEST_BASE_SOURCES} ${UNIT_TEST_CUDA_SOURCES})
target_link_libraries(unit_test_cuda PRIVATE Catch2::Catch2 StencilStream_CUDA)

add_executable(unit_test_fpga ${UNIT_TEST_BASE_SOURCES} ${UNIT_TEST_FPGA_SOURCES})
target_link_libraries(unit_test_fpga PRIVATE Catch2::Catch2 StencilStream_MonotileEmulator StencilStream_TilingEmulator)
