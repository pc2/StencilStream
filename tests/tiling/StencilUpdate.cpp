/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
 * University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "../HostPipe.hpp"
#include "../StencilUpdateTest.hpp"
#include "../TransFuncs.hpp"
#include <StencilStream/BaseTransitionFunction.hpp>
#include <StencilStream/tiling/StencilUpdate.hpp>
#include <catch2/catch_all.hpp>

using namespace sycl;
using namespace stencil;
using namespace stencil::tiling;

template <std::size_t stencil_radius, std::size_t temporal_parallelism,
          std::size_t spatial_parallelism, std::size_t tile_height, std::size_t tile_width>
void test_tiling_kernel(std::size_t grid_height, std::size_t grid_width, std::size_t tile_r,
                        std::size_t tile_c, std::size_t iteration_offset,
                        std::size_t target_i_iteration) {
    using TransFunc = FPGATransFunc<stencil_radius>;
    using CellVector = std::array<typename TransFunc::Cell, spatial_parallelism>;
    using in_pipe = sycl::pipe<class TilingExecutionKernelInPipeID, CellVector>;
    using out_pipe = sycl::pipe<class TilingExecutionKernelOutPipeID, CellVector>;
    using TDVGlobalState =
        tdv::single_pass::InlineStrategy::template GlobalState<TransFunc, temporal_parallelism>;
    using TDVKernelArgument = typename TDVGlobalState::KernelArgument;
    using TestExecutionKernel =
        StencilUpdateKernel<TransFunc, TDVKernelArgument, temporal_parallelism, spatial_parallelism,
                            tile_height, tile_width, in_pipe, out_pipe>;

    constexpr std::size_t n_processing_elements = temporal_parallelism * TransFunc::n_subiterations;
    constexpr std::size_t stencil_buffer_lead =
        int_ceil_div(stencil_radius, spatial_parallelism) * spatial_parallelism;
    constexpr std::size_t halo_height = stencil_radius * n_processing_elements;
    constexpr std::size_t halo_width = stencil_buffer_lead * n_processing_elements;
    // Tile width has to be multiple of vector length.
    constexpr std::size_t vect_tile_width = tile_width / spatial_parallelism;

    std::size_t output_tile_section_height =
        std::min(tile_height, grid_height - tile_r * tile_height);
    std::size_t vect_output_tile_section_width = std::min(
        vect_tile_width, int_ceil_div(grid_width - tile_c * tile_width, spatial_parallelism));

    sycl::queue working_queue;

    working_queue.submit([&](sycl::handler &cgh) {
        // Using negative numbers here to make life easier.
        int start_row = tile_r * tile_height - halo_height;
        int end_row = start_row + output_tile_section_height + 2 * halo_height;
        int vect_start_column = tile_c * vect_tile_width - (halo_width / spatial_parallelism);
        int vect_end_column = vect_start_column + vect_output_tile_section_width +
                              2 * (halo_width / spatial_parallelism);

        cgh.single_task([=]() {
            for (int r = start_row; r < end_row; r++) {
                for (int vect_c = vect_start_column; vect_c < vect_end_column; vect_c++) {
                    CellVector input_vector;

#pragma unroll
                    for (int i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                        int c = vect_c * spatial_parallelism + i_cell;
                        if (r >= 0 && c >= 0 && r < grid_height && c < grid_width) {
                            input_vector[i_cell] =
                                Cell{r, c, int(iteration_offset), 0, CellStatus::Normal};
                        } else {
                            input_vector[i_cell] = Cell::halo();
                        }
                    }

                    in_pipe::write(input_vector);
                }
            }
        });
    });

    TDVGlobalState global_state(TransFunc(), iteration_offset, target_i_iteration);
    working_queue.submit([&](sycl::handler &cgh) {
        TDVKernelArgument kernel_argument(global_state, cgh, iteration_offset, target_i_iteration);
        TestExecutionKernel kernel(TransFunc(), iteration_offset, target_i_iteration,
                                   tile_r * tile_height, tile_c * tile_width, grid_height,
                                   grid_width, Cell::halo(), kernel_argument);
        cgh.single_task(kernel);
    });

    buffer<CellVector, 2> output_buffer(
        range<2>(output_tile_section_height, vect_output_tile_section_width));

    working_queue.submit([&](sycl::handler &cgh) {
        accessor output_buffer_ac(output_buffer, cgh, read_write);
        cgh.single_task([=]() {
            for (std::size_t r = 0; r < output_buffer_ac.get_range()[0]; r++) {
                for (std::size_t vect_c = 0; vect_c < output_buffer_ac.get_range()[1]; vect_c++) {
                    output_buffer_ac[r][vect_c] = out_pipe::read();
                }
            }
        });
    });

    host_accessor output_buffer_ac(output_buffer, read_only);
    for (std::size_t local_r = 0; local_r < output_buffer_ac.get_range()[0]; local_r++) {
        for (std::size_t vect_local_c = 0; vect_local_c < output_buffer_ac.get_range()[1];
             vect_local_c++) {
            for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                std::size_t r = tile_r * tile_height + local_r;
                std::size_t c = tile_c * tile_width + vect_local_c * spatial_parallelism + i_cell;

                Cell cell = output_buffer_ac[local_r][vect_local_c][i_cell];
                if (c < grid_width) {
                    REQUIRE(cell.r == r);
                    REQUIRE(cell.c == c);
                    REQUIRE(cell.status == CellStatus::Normal);
                    REQUIRE(cell.i_iteration == target_i_iteration);
                    REQUIRE(cell.i_subiteration == 0);
                } else {
                    REQUIRE(cell.r == 0);
                    REQUIRE(cell.c == 0);
                    REQUIRE(cell.status == CellStatus::Halo);
                    REQUIRE(cell.i_iteration == 0);
                    REQUIRE(cell.i_subiteration == 0);
                }
            }
        }
    }
}

template <std::size_t stencil_radius, std::size_t temporal_parallelism,
          std::size_t spatial_parallelism, std::size_t tile_height, std::size_t tile_width>
void test_tiling_kernel_on_grid() {
    auto test_tiling_kernel_impl =
        &test_tiling_kernel<stencil_radius, temporal_parallelism, spatial_parallelism, tile_height,
                            tile_width>;
    // All possible tile types (corner, border, core tiles), where each tile is completely filled.
    test_tiling_kernel_impl(3 * tile_height, 3 * tile_width, 0, 0, 0, temporal_parallelism);
    test_tiling_kernel_impl(3 * tile_height, 3 * tile_width, 0, 1, 0, temporal_parallelism);
    test_tiling_kernel_impl(3 * tile_height, 3 * tile_width, 0, 2, 0, temporal_parallelism);
    test_tiling_kernel_impl(3 * tile_height, 3 * tile_width, 1, 0, 0, temporal_parallelism);
    test_tiling_kernel_impl(3 * tile_height, 3 * tile_width, 1, 1, 0, temporal_parallelism);
    test_tiling_kernel_impl(3 * tile_height, 3 * tile_width, 1, 2, 0, temporal_parallelism);
    test_tiling_kernel_impl(3 * tile_height, 3 * tile_width, 2, 0, 0, temporal_parallelism);
    test_tiling_kernel_impl(3 * tile_height, 3 * tile_width, 2, 1, 0, temporal_parallelism);
    test_tiling_kernel_impl(3 * tile_height, 3 * tile_width, 2, 2, 0, temporal_parallelism);

    // Missing bottom row.
    test_tiling_kernel_impl(tile_height - 1, tile_width, 0, 0, 0, temporal_parallelism);
    // Missing right column.
    test_tiling_kernel_impl(tile_height, tile_width - 1, 0, 0, 0, temporal_parallelism);
    // Both missing bottom row and right column.
    test_tiling_kernel_impl(tile_height - 1, tile_width - 1, 0, 0, 0, temporal_parallelism);

    // Only a single iteration.
    test_tiling_kernel_impl(tile_height, tile_width, 0, 0, 0, 1);
    // Second pass.
    test_tiling_kernel_impl(tile_height, tile_width, 0, 0, temporal_parallelism,
                            2 * temporal_parallelism);
    // A single iteration for the second pass.
    test_tiling_kernel_impl(tile_height, tile_width, 0, 0, temporal_parallelism,
                            temporal_parallelism + 1);
}

TEST_CASE("tiling::StencilUpdateKernel", "[tiling::StencilUpdateKernel]") {
    // Radius of 1, two PEs (1x), no vectorization
    test_tiling_kernel_on_grid<1, 2, 1, 32, 16>();
    // Radius of 1, four PEs (2x), no vectorization
    test_tiling_kernel_on_grid<1, 4, 1, 32, 16>();
    // Radius of 1, eight PEs (4x), no vectorization
    test_tiling_kernel_on_grid<1, 8, 1, 32, 16>();

    // Radius of 1, two PEs (1x), 2x vectorization
    test_tiling_kernel_on_grid<1, 2, 2, 32, 16>();
    // Radius of 1, four PEs (2x), 2x vectorization
    test_tiling_kernel_on_grid<1, 4, 2, 32, 16>();
    // Radius of 1, eight PEs (4x), 2x vectorization
    test_tiling_kernel_on_grid<1, 8, 2, 32, 16>();

    // Radius of 1, two PEs (1x), 4x vectorization
    test_tiling_kernel_on_grid<1, 2, 4, 32, 16>();
    // Radius of 1, four PEs (2x), 4x vectorization
    test_tiling_kernel_on_grid<1, 4, 4, 32, 16>();
    // Radius of 1, eight PEs (4x), 4x vectorization
    test_tiling_kernel_on_grid<1, 8, 4, 32, 16>();
}

TEST_CASE("tiling::StencilUpdate", "[tiling::StencilUpdate]") {
    constexpr std::size_t temporal_parallelism = 2;
    constexpr std::size_t tile_height = 128;
    constexpr std::size_t tile_width = 64;

    using StencilUpdateVect1Impl =
        StencilUpdate<FPGATransFunc<1>, temporal_parallelism, 1, tile_height, tile_width>;
    using GridVect1Impl = Grid<FPGATransFunc<1>::Cell, 1>;

    using StencilUpdateVect2Impl =
        StencilUpdate<FPGATransFunc<1>, temporal_parallelism, 2, tile_height, tile_width>;
    using GridVect2Impl = Grid<FPGATransFunc<1>::Cell, 2>;

    using StencilUpdateVect4Impl =
        StencilUpdate<FPGATransFunc<1>, temporal_parallelism, 4, tile_height, tile_width>;
    using GridVect4Impl = Grid<FPGATransFunc<1>::Cell, 4>;

    static_assert(concepts::StencilUpdate<StencilUpdateVect1Impl, FPGATransFunc<1>, GridVect1Impl>);
    static_assert(concepts::StencilUpdate<StencilUpdateVect2Impl, FPGATransFunc<1>, GridVect2Impl>);
    static_assert(concepts::StencilUpdate<StencilUpdateVect4Impl, FPGATransFunc<1>, GridVect4Impl>);

    for (std::size_t i_grid_height = 0; i_grid_height < 3; i_grid_height++) {
        for (std::size_t i_grid_width = 0; i_grid_width < 3; i_grid_width++) {
            std::size_t grid_height = (1 + i_grid_height) * (tile_height / 2);
            std::size_t grid_width = (1 + i_grid_width) * (tile_width / 2);

            test_stencil_update<GridVect1Impl, StencilUpdateVect1Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect1Impl, StencilUpdateVect1Impl>(grid_height, grid_width, 1,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect1Impl, StencilUpdateVect1Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism + 1);
            test_stencil_update<GridVect1Impl, StencilUpdateVect1Impl>(grid_height - 1, grid_width,
                                                                       0, temporal_parallelism + 1);
            test_stencil_update<GridVect1Impl, StencilUpdateVect1Impl>(grid_height, grid_width - 1,
                                                                       0, temporal_parallelism + 1);

            test_stencil_update<GridVect2Impl, StencilUpdateVect2Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect2Impl, StencilUpdateVect2Impl>(grid_height, grid_width, 1,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect2Impl, StencilUpdateVect2Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism + 1);
            test_stencil_update<GridVect2Impl, StencilUpdateVect2Impl>(grid_height - 1, grid_width,
                                                                       0, temporal_parallelism + 1);
            test_stencil_update<GridVect2Impl, StencilUpdateVect2Impl>(grid_height, grid_width - 1,
                                                                       0, temporal_parallelism + 1);

            test_stencil_update<GridVect4Impl, StencilUpdateVect4Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4Impl>(grid_height, grid_width, 1,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism + 1);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4Impl>(grid_height - 1, grid_width,
                                                                       0, temporal_parallelism + 1);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4Impl>(grid_height, grid_width - 1,
                                                                       0, temporal_parallelism + 1);
        }
    }
}