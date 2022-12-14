set(UNIT_TEST_SOURCES
    HostPipe.cpp
    Stencil.cpp
    StencilExecutor.cpp
    TimeDependentValueSupplier.cpp
    monotile/ExecutionKernel.cpp
    tiling/ExecutionKernel.cpp
    tiling/Grid.cpp
    tiling/Tile.cpp
)

add_executable(unit_test ${UNIT_TEST_SOURCES})
target_link_libraries(unit_test PRIVATE Catch2::Catch2WithMain)
