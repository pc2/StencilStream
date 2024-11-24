# Pure CPU-based executable
add_executable(diffusion diffusion.cpp)
target_link_libraries(diffusion PUBLIC StencilStream_CPU)