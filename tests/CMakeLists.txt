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
  GIT_TAG        v3.3.2
)
FetchContent_MakeAvailable(Catch2)

add_compile_definitions(STENCIL_INDEX_WIDTH=32)

set(UNIT_TEST_SOURCES
    HostPipe.cpp
    Stencil.cpp
    cpu/Grid.cpp
    cpu/StencilUpdate.cpp
    monotile/Grid.cpp
    monotile/StencilUpdate.cpp
    tiling/Grid.cpp
    tiling/StencilUpdate.cpp
    constraint_test.cpp
)

add_executable(unit_test ${UNIT_TEST_SOURCES})
target_link_libraries(unit_test PRIVATE Catch2::Catch2WithMain StencilStream_TilingEmulator StencilStream_MonotileEmulator)
