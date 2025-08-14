# Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

add_executable(jacobi jacobi.cpp)
target_link_libraries(jacobi PUBLIC StencilStream_CPU)
target_compile_definitions(jacobi PUBLIC JACOBI_KERNEL=Jacobi4General)

add_custom_target(jacobi_reports)

foreach(KERNEL Jacobi1General Jacobi2Constant Jacobi3Constant Jacobi4Constant Jacobi5Constant Jacobi4General Jacobi5General Jacobi9General)
    foreach(BACKEND mono mono_emu mono_report tiling tiling_emu tiling_report)

        set(EXECUTABLE "${KERNEL}_${BACKEND}")
        add_executable(${EXECUTABLE} jacobi.cpp)
        target_compile_definitions(${EXECUTABLE} PUBLIC JACOBI_KERNEL=${KERNEL})

        if(${BACKEND} MATCHES "^mono")
            if(${BACKEND} MATCHES "emu")
                target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_MonotileEmulator)
            elseif(${BACKEND} MATCHES "report")
                target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_MonotileReport)
            else()
                target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_Monotile)
            endif()
            target_link_options(${EXECUTABLE} PUBLIC -Xsclock=350MHz -Xsprofile)
            target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_NamedKernels)
        elseif(${BACKEND} MATCHES "^tiling")
            if(${BACKEND} MATCHES "emu")
                target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_TilingEmulator)
            elseif(${BACKEND} MATCHES "report")
                target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_TilingReport)
            else()
                target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_Tiling)
            endif()
            target_link_options(${EXECUTABLE} PUBLIC -Xsclock=350MHz -Xsprofile)
            target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_NamedKernels)
        endif()

        if(${BACKEND} MATCHES "report")
            add_dependencies(jacobi_reports ${EXECUTABLE})
        endif()
    endforeach()
endforeach()

target_link_options(Jacobi2Constant_mono PUBLIC -Xsseed=10893)
target_link_options(Jacobi2Constant_tiling PUBLIC -Xsseed=10893)
target_link_options(Jacobi4Constant_mono PUBLIC -Xsseed=10893)
target_link_options(Jacobi5Constant_mono PUBLIC -Xsseed=10594)
target_link_options(Jacobi5Constant_tiling PUBLIC -Xsseed=10893)
target_link_options(Jacobi4General_tiling PUBLIC -Xsseed=10893)
target_link_options(Jacobi4General_mono PUBLIC -Xsseed=10893)
target_link_options(Jacobi5General_tiling PUBLIC -Xsseed=10893)
target_link_options(Jacobi9General_mono PUBLIC -Xsseed=25143)
target_link_options(Jacobi9General_tiling PUBLIC -Xsseed=25547)