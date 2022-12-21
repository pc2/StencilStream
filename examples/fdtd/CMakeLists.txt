# Copyright © 2020-2022Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

include_directories(.)
add_compile_options(-std=c++20 -qactypes)
add_compile_definitions(STENCIL_INDEX_WIDTH=32)

add_custom_target(fdtd_reports)

foreach(MATERIAL coef lut render)
    foreach(TDVS inline device host)
        foreach(EXECUTOR mono mono_emu mono_report tiling tiling_emu tiling_report cpu)
            set(EXECUTABLE "fdtd_${MATERIAL}_${TDVS}_${EXECUTOR}")
            add_executable(${EXECUTABLE} src/fdtd.cpp)

            if(${MATERIAL} STREQUAL coef)
                target_compile_definitions(${EXECUTABLE} PUBLIC MATERIAL=0)
            elseif(${MATERIAL} STREQUAL lut)
                target_compile_definitions(${EXECUTABLE} PUBLIC MATERIAL=1)
            elseif(${MATERIAL} STREQUAL render)
                target_compile_definitions(${EXECUTABLE} PUBLIC MATERIAL=2)
            endif()

            if(${TDVS} STREQUAL "inline")
                target_compile_definitions(${EXECUTABLE} PUBLIC TDVS_TYPE=0)
            elseif(${TDVS} STREQUAL "device")
                target_compile_definitions(${EXECUTABLE} PUBLIC TDVS_TYPE=1)
            elseif(${TDVS} STREQUAL "host")
                target_compile_definitions(${EXECUTABLE} PUBLIC TDVS_TYPE=2)
            endif()

            if(${EXECUTOR} MATCHES "^mono")
                target_compile_definitions(${EXECUTABLE} PUBLIC EXECUTOR=0)
            elseif(${EXECUTOR} MATCHES "^tiling")
                target_compile_definitions(${EXECUTABLE} PUBLIC EXECUTOR=1)
            elseif(${EXECUTOR} STREQUAL "cpu")
                target_compile_definitions(${EXECUTABLE} PUBLIC EXECUTOR=2)
            endif()

            if(NOT (${EXECUTOR} STREQUAL "cpu"))
                target_compile_options(${EXECUTABLE} PUBLIC -fintelfpga)
                target_link_options(${EXECUTABLE} PUBLIC -fintelfpga)

                if(NOT (${EXECUTOR} MATCHES "emu"))
                    target_compile_definitions(${EXECUTABLE} PUBLIC HARDWARE)
                    target_link_options(${EXECUTABLE} PUBLIC -reuse-exe=${EXECUTABLE} -Xshardware)

                    if(VERBOSE_SYNTHESIS)
                        target_link_options(${EXECUTABLE} PUBLIC -Xsv)
                    endif()
                endif()

                if(${EXECUTOR} MATCHES "report")
                    target_link_options(${EXECUTABLE} PUBLIC -fsycl-link)
                    add_dependencies(fdtd_reports ${EXECUTABLE})
                endif()
            endif()
        endforeach()
    endforeach()
endforeach()