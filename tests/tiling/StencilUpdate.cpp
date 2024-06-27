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
#include "../constants.hpp"
#include <StencilStream/BaseTransitionFunction.hpp>
#include <StencilStream/tiling/StencilUpdate.hpp>
#include <catch2/catch_all.hpp>

using namespace sycl;
using namespace stencil;
using namespace stencil::tiling;

template <
    tdv::single_pass::Strategy<FPGATransFunc<stencil_radius>, n_processing_elements> TDVStrategy>
void test_tiling_kernel_with_strategy(uindex_t grid_width, uindex_t grid_height,
                                      uindex_t iteration_offset, uindex_t target_i_iteration) {
    using TransFunc = FPGATransFunc<stencil_radius>;
    using in_pipe = sycl::pipe<class TilingExecutionKernelInPipeID, Cell>;
    using out_pipe = sycl::pipe<class TilingExecutionKernelOutPipeID, Cell>;
    using TDVGlobalState = TDVStrategy::template GlobalState<TransFunc, n_processing_elements>;
    using TDVKernelArgument = typename TDVGlobalState::KernelArgument;
    using TestExecutionKernel =
        StencilUpdateKernel<TransFunc, TDVKernelArgument, n_processing_elements, tile_width,
                            tile_height, in_pipe, out_pipe>;

    sycl::queue working_queue;

    working_queue.submit([&](sycl::handler &cgh) {
        cgh.single_task([=]() {
            for (index_t c = -halo_radius; c < index_t(halo_radius + grid_width); c++) {
                for (index_t r = -halo_radius; r < index_t(halo_radius + grid_height); r++) {
                    if (c >= index_t(0) && c < index_t(grid_width) && r >= index_t(0) &&
                        r < index_t(grid_height)) {
                        in_pipe::write(
                            Cell{c, r, index_t(iteration_offset), 0, CellStatus::Normal});
                    } else {
                        in_pipe::write(Cell::halo());
                    }
                }
            }
        });
    });

    TDVGlobalState global_state(TransFunc(), iteration_offset, target_i_iteration);
    working_queue.submit([&](sycl::handler &cgh) {
        TDVKernelArgument kernel_argument(global_state, cgh, iteration_offset, target_i_iteration);
        TestExecutionKernel kernel(TransFunc(), iteration_offset, target_i_iteration, 0, 0,
                                   grid_width, grid_height, Cell::halo(), kernel_argument);
        cgh.single_task(kernel);
    });

    buffer<Cell, 2> output_buffer(range<2>(grid_width, grid_height));

    working_queue.submit([&](sycl::handler &cgh) {
        accessor output_buffer_ac(output_buffer, cgh, read_write);
        cgh.single_task([=]() {
            for (uindex_t c = 0; c < grid_width; c++) {
                for (uindex_t r = 0; r < grid_height; r++) {
                    output_buffer_ac[c][r] = out_pipe::read();
                }
            }
        });
    });

    host_accessor output_buffer_ac(output_buffer, read_only);
    for (uindex_t c = 1; c < grid_width; c++) {
        for (uindex_t r = 1; r < grid_height; r++) {
            Cell cell = output_buffer_ac[c][r];
            REQUIRE(cell.c == c);
            REQUIRE(cell.r == r);
            REQUIRE(cell.i_iteration == target_i_iteration);
            REQUIRE(cell.i_subiteration == 0);
            REQUIRE(cell.status == CellStatus::Normal);
        }
    }
}

void test_tiling_kernel(uindex_t grid_width, uindex_t grid_height, uindex_t iteration_offset,
                        uindex_t target_i_iteration) {
    test_tiling_kernel_with_strategy<tdv::single_pass::InlineStrategy>(
        tile_width, tile_height, iteration_offset, target_i_iteration);
    test_tiling_kernel_with_strategy<tdv::single_pass::PrecomputeOnDeviceStrategy>(
        tile_width, tile_height, iteration_offset, target_i_iteration);
    test_tiling_kernel_with_strategy<tdv::single_pass::PrecomputeOnHostStrategy>(
        tile_width, tile_height, iteration_offset, target_i_iteration);
}

TEST_CASE("tiling::StencilUpdateKernel", "[tiling::StencilUpdateKernel]") {
    test_tiling_kernel(tile_width, tile_height, 0, iters_per_pass);
}

TEST_CASE("tiling::StencilUpdateKernel (partial tile)", "[tiling::StencilUpdateKernel]") {
    test_tiling_kernel(tile_width / 2, tile_height, 0, iters_per_pass);
}

TEST_CASE("tiling::StencilUpdateKernel (partial pipeline)", "[tiling::StencilUpdateKernel]") {
    static_assert(iters_per_pass != 1);
    test_tiling_kernel(tile_width, tile_height, 0, iters_per_pass - 1);
}

TEST_CASE("tiling::StencilUpdateKernel (iteration offset)", "[tiling::StencilUpdateKernel]") {
    test_tiling_kernel(tile_width, tile_height, iters_per_pass, 2 * iters_per_pass);
}

TEST_CASE("tiling::StencilUpdateKernel (iteration offset, partial pipeline)",
          "[tiling::StencilUpdateKernel]") {
    static_assert(iters_per_pass != 1);
    test_tiling_kernel(tile_width, tile_height, iters_per_pass, 2 * iters_per_pass - 1);
}

struct HaloHandlingKernel : public BaseTransitionFunction {
    using Cell = bool;

    bool operator()(Stencil<bool, 1> const &stencil) const {
        ID idx = stencil.id;
        bool is_valid = true;
        if (idx.c == 0) {
            is_valid &= stencil[ID(-1, -1)] && stencil[ID(-1, 0)] && stencil[ID(-1, 1)];
        } else if (idx.c == tile_width - 1) {
            is_valid &= stencil[ID(1, -1)] && stencil[ID(1, 0)] && stencil[ID(1, 1)];
        }

        if (idx.r == 0) {
            is_valid &= stencil[ID(-1, -1)] && stencil[ID(0, -1)] && stencil[ID(1, -1)];
        } else if (idx.r == tile_height - 1) {
            is_valid &= stencil[ID(-1, 1)] && stencil[ID(0, 1)] && stencil[ID(1, 1)];
        }

        return is_valid;
    }
};

TEST_CASE("Halo values inside the pipeline are handled correctly",
          "[tiling::StencilUpdateKernel]") {
    using in_pipe = sycl::pipe<class HaloValueTestInPipeID, bool>;
    using out_pipe = sycl::pipe<class HaloValueTestOutPipeID, bool>;
    using TDVGlobalState =
        tdv::single_pass::InlineStrategy::template GlobalState<HaloHandlingKernel,
                                                               n_processing_elements>;
    using TDVKernelArgument = typename TDVGlobalState::KernelArgument;
    using TestExecutionKernel =
        StencilUpdateKernel<HaloHandlingKernel, TDVKernelArgument, n_processing_elements,
                            tile_width, tile_height, in_pipe, out_pipe>;

    sycl::queue working_queue;

    uindex_t halo_radius = n_processing_elements;

    working_queue.submit([&](sycl::handler &cgh) {
        cgh.single_task([=]() {
            for (index_t c = -halo_radius; c < index_t(halo_radius + tile_width); c++) {
                for (index_t r = -halo_radius; r < index_t(halo_radius + tile_height); r++) {
                    in_pipe::write(false);
                }
            }
        });
    });

    TDVGlobalState global_state(HaloHandlingKernel(), 0, iters_per_pass);
    working_queue.submit([&](sycl::handler &cgh) {
        TDVKernelArgument kernel_argument(global_state, cgh, 0, iters_per_pass);
        TestExecutionKernel kernel(HaloHandlingKernel(), 0, iters_per_pass, 0, 0, tile_width,
                                   tile_height, true, kernel_argument);
        cgh.single_task(kernel);
    });

    sycl::buffer<bool, 2> is_correct(sycl::range<2>(tile_width, tile_height));
    working_queue.submit([&](sycl::handler &cgh) {
        sycl::accessor is_correct_ac(is_correct, cgh);
        cgh.single_task([=]() {
            for (uindex_t c = 0; c < tile_width; c++) {
                for (uindex_t r = 0; r < tile_height; r++) {
                    is_correct_ac[c][r] = out_pipe::read();
                }
            }
        });
    });

    sycl::host_accessor is_correct_ac(is_correct);
    for (uindex_t c = 0; c < tile_width; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            REQUIRE(is_correct_ac[c][r]);
        }
    }
}

using StencilUpdateImpl =
    StencilUpdate<FPGATransFunc<1>, n_processing_elements, tile_width, tile_height>;
using GridImpl = typename StencilUpdateImpl::GridImpl;

static_assert(concepts::StencilUpdate<StencilUpdateImpl, FPGATransFunc<1>, GridImpl>);

TEST_CASE("tiling::StencilUpdate", "[tiling::StencilUpdate]") {
    for (uindex_t i_grid_width = 0; i_grid_width < 3; i_grid_width++) {
        for (uindex_t i_grid_height = 0; i_grid_height < 3; i_grid_height++) {
            uindex_t grid_width = (1 + i_grid_width) * (tile_width / 2);
            uindex_t grid_height = (1 + i_grid_height) * (tile_height / 2);

            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_width, grid_height, 0,
                                                             iters_per_pass);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_width, grid_height, 1,
                                                             iters_per_pass);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_width, grid_height, 0,
                                                             iters_per_pass + 1);
        }
    }
}