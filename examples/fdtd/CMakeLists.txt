# Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

add_compile_definitions(STENCIL_INDEX_WIDTH=32)

include(FetchContent)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

add_custom_target(fdtd_reports)

foreach(MATERIAL coef lut render)
    foreach(TDVS inline device host)
        foreach(EXECUTOR mono mono_emu mono_report tiling tiling_emu tiling_report cpu)
            set(EXECUTABLE "fdtd_${MATERIAL}_${TDVS}_${EXECUTOR}")
            add_executable(${EXECUTABLE} src/fdtd.cpp)
            target_link_libraries(${EXECUTABLE} PRIVATE nlohmann_json::nlohmann_json)

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
                if(${EXECUTOR} MATCHES "emu")
                    target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_MonotileEmulator)
                elseif(${EXECUTOR} MATCHES "report")
                    target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_MonotileReport)
                else()
                    target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_Monotile)
                endif()
            elseif(${EXECUTOR} MATCHES "^tiling")
                if(${EXECUTOR} MATCHES "emu")
                    target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_TilingEmulator)
                elseif(${EXECUTOR} MATCHES "report")
                    target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_TilingReport)
                else()
                    target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_Tiling)
                endif()
            elseif(${EXECUTOR} STREQUAL "cpu")
                target_link_libraries(${EXECUTABLE} PUBLIC StencilStream_CPU)
            endif()

            if(${EXECUTOR} MATCHES "tiling")
                target_link_options(${EXECUTABLE} PUBLIC -Xsclock=300MHz)
            endif()
        endforeach()
    endforeach()
endforeach()