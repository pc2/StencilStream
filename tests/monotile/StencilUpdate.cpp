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
#include <StencilStream/monotile/StencilUpdate.hpp>
#include <catch2/catch_all.hpp>

using namespace sycl;
using namespace stencil;
using namespace stencil::monotile;

void test_monotile_kernel(uindex_t grid_width, uindex_t grid_height, uindex_t iteration_offset,
                          uindex_t target_i_iteration) {
    using TransFunc = FPGATransFunc<stencil_radius>;
    using in_pipe = sycl::pipe<class MonotileExecutionKernelInPipeID, Cell>;
    using out_pipe = sycl::pipe<class MonotileExecutionKernelOutPipeID, Cell>;
    using GlobalState =
        tdv::single_pass::InlineStrategy::GlobalState<TransFunc, n_processing_elements>;
    using KernelArgument = typename GlobalState::KernelArgument;
    using TestExecutionKernel =
        StencilUpdateKernel<TransFunc, KernelArgument, n_processing_elements, tile_width,
                            tile_height, in_pipe, out_pipe>;

    sycl::queue working_queue;

    working_queue.submit([&](sycl::handler &cgh) {
        cgh.single_task([=]() {
            for (uindex_t c = 0; c < grid_width; c++) {
                for (uindex_t r = 0; r < grid_height; r++) {
                    in_pipe::write(Cell{index_t(c), index_t(r), index_t(iteration_offset), 0,
                                        CellStatus::Normal});
                }
            }
        });
    });

    GlobalState global_state(TransFunc(), iteration_offset, target_i_iteration);
    working_queue.submit([&](sycl::handler &cgh) {
        KernelArgument kernel_argument(global_state, cgh, iteration_offset, target_i_iteration);

        cgh.single_task(TestExecutionKernel(TransFunc(), iteration_offset, target_i_iteration,
                                            grid_width, grid_height, Cell::halo(),
                                            kernel_argument));
    });

    buffer<Cell, 2> output_buffer(range<2>(grid_width, grid_height));

    working_queue.submit([&](sycl::handler &cgh) {
        accessor output_buffer_ac(output_buffer, cgh, write_only);
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

TEST_CASE("monotile::StencilUpdateKernel", "[monotile::StencilUpdateKernel]") {
    test_monotile_kernel(tile_width, tile_height, 0, iters_per_pass);
}

TEST_CASE("monotile::StencilUpdateKernel (partial tile)", "[monotile::StencilUpdateKernel]") {
    test_monotile_kernel(tile_width / 2, tile_height / 2, 0, iters_per_pass);
}

TEST_CASE("monotile::StencilUpdateKernel (partial pipeline)", "[monotile::StencilUpdateKernel]") {
    static_assert(iters_per_pass != 1);
    test_monotile_kernel(tile_width, tile_height, 0, iters_per_pass - 1);
}

TEST_CASE("monotile::StencilUpdateKernel (noop)", "[monotile::StencilUpdateKernel]") {
    test_monotile_kernel(tile_width, tile_height, 0, 0);
}

TEST_CASE("monotile::StencilUpdateKernel (incomplete pipeline, offset != 0)",
          "[monotile::StencilUpdateKernel]") {
    test_monotile_kernel(tile_width, tile_height, iters_per_pass / 2, iters_per_pass);
}

template <typename TDVStrategy> void test_monotile_update() {
    using StencilUpdateImpl = StencilUpdate<FPGATransFunc<1>, n_processing_elements, tile_width,
                                            tile_height, TDVStrategy>;
    using GridImpl = Grid<Cell>;
    static_assert(concepts::StencilUpdate<StencilUpdateImpl, FPGATransFunc<1>, GridImpl>);

    for (uindex_t grid_width = tile_width / 2; grid_width < tile_width; grid_width += 1) {
        for (uindex_t grid_height = tile_height / 2; grid_height < tile_height; grid_height += 1) {
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_width, grid_height, 0,
                                                             iters_per_pass);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_width, grid_height, 1,
                                                             iters_per_pass);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_width, grid_height, 0,
                                                             iters_per_pass + 1);
        }
    }
}

TEST_CASE("monotile::StencilUpdate", "[monotile::StencilUpdate]") {
    test_monotile_update<tdv::single_pass::InlineStrategy>();
    test_monotile_update<tdv::single_pass::PrecomputeOnDeviceStrategy>();
    test_monotile_update<tdv::single_pass::PrecomputeOnHostStrategy>();
}