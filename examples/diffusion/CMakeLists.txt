# Pure GPU-based executable
add_executable(diffusion_gpu diffusion.cpp)
target_link_libraries(diffusion_gpu PUBLIC StencilStream_GPU)

# Pure CPU-based executable
add_executable(diffusion_cpu diffusion.cpp)
target_link_libraries(diffusion_cpu PUBLIC StencilStream_CPU)