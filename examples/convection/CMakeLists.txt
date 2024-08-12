# Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Global settings
add_compile_definitions(STENCIL_INDEX_WIDTH=32)

add_executable(convection_cpu convection.cpp)
target_link_libraries(convection_cpu PUBLIC StencilStream_CPU)

add_executable(convection convection.cpp)
target_link_libraries(convection PUBLIC StencilStream_Monotile)

add_executable(convection_emu convection.cpp)
target_link_libraries(convection_emu PUBLIC StencilStream_MonotileEmulator)

add_executable(convection_report convection.cpp)
target_link_libraries(convection_report PUBLIC StencilStream_MonotileReport)

# Setting hardware synthesis options
foreach(TARGET convection convection_report)
    target_link_options(${TARGET} PUBLIC -Xsseed=63764)
endforeach()

include(FetchContent)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Linking with the JSON library
foreach(TARGET convection convection_cpu convection_emu convection_report)
    target_link_libraries(${TARGET} PRIVATE nlohmann_json::nlohmann_json)
endforeach()