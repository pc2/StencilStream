# Pure CPU-based executable
add_executable(diffusion_cpu diffusion.cpp)
target_link_libraries(diffusion_cpu PUBLIC StencilStream_CPU)

# Pure GPU-based executable
add_executable(diffusion_cuda diffusion.cpp)
target_link_libraries(diffusion_cuda PUBLIC StencilStream_CUDA)