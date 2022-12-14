set(CMAKE_CXX_COMPILER dpcpp)
cmake_minimum_required(VERSION 3.21)

project(StencilStream CXX)
set(CMAKE_PROJECT_VERSION 2.1.1)

find_package(Boost 1.74.0 REQUIRED)

Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.2.1
)
FetchContent_MakeAvailable(Catch2)

include_directories(.)
add_compile_options(-std=c++20 -qactypes -fintelfpga -Wtautological-constant-compare)
add_link_options(-std=c++20 -qactypes -fintelfpga -Wtautological-constant-compare)
add_compile_definitions(STENCIL_INDEX_WIDTH=32)

add_subdirectory(tests)