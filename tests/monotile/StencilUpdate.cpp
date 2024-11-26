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

constexpr std::size_t my_vector_length = 4;

void test_monotile_kernel(std::size_t grid_height, std::size_t grid_width,
                          std::size_t iteration_offset, std::size_t target_i_iteration) {
    using TransFunc = HostTransFunc<stencil_radius>;
    using in_pipe =
        HostPipe<class MonotileExecutionKernelInPipeID, std::array<Cell, my_vector_length>>;
    using out_pipe =
        HostPipe<class MonotileExecutionKernelOutPipeID, std::array<Cell, my_vector_length>>;
    using GlobalState = tdv::single_pass::InlineStrategy::GlobalState<TransFunc, 1>;
    using KernelArgument = typename GlobalState::KernelArgument;
    using TestExecutionKernel =
        StencilUpdateKernel<TransFunc, KernelArgument, n_processing_elements, my_vector_length,
                            tile_height, tile_width, in_pipe, out_pipe>;

    sycl::queue working_queue;

    for (std::size_t r = 0; r < grid_height; r++) {
        for (std::size_t c = 0; c < grid_width; c += my_vector_length) {
            std::array<Cell, my_vector_length> vector;
            for (std::size_t i_cell = 0; i_cell < my_vector_length; i_cell++) {
                vector[i_cell] =
                    Cell{int(r), int(c + i_cell), int(iteration_offset), 0, CellStatus::Normal};
            }
            in_pipe::write(vector);
        }
    }

    GlobalState global_state(TransFunc(), iteration_offset, target_i_iteration);
    KernelArgument kernel_argument(global_state, iteration_offset);

    TestExecutionKernel kernel(TransFunc(), iteration_offset, target_i_iteration, grid_height,
                               grid_width, Cell::halo(), kernel_argument);
    kernel();

    for (std::size_t r = 0; r < grid_height; r++) {
        for (std::size_t c = 0; c < grid_width; c += my_vector_length) {
            std::array<Cell, my_vector_length> vector = out_pipe::read();
            for (std::size_t i_cell = 0; i_cell < my_vector_length; i_cell++) {
                Cell cell = vector[i_cell];
                REQUIRE(cell.r == r);
                REQUIRE(cell.c == c + i_cell);
                REQUIRE(cell.i_iteration == target_i_iteration);
                REQUIRE(cell.i_subiteration == 0);
                REQUIRE(cell.status == CellStatus::Normal);
            }
        }
    }
}

TEST_CASE("monotile::StencilUpdateKernel", "[monotile::StencilUpdateKernel]") {
    test_monotile_kernel(tile_height, tile_width, 0, iters_per_pass);
}

TEST_CASE("monotile::StencilUpdateKernel (partial tile)", "[monotile::StencilUpdateKernel]") {
    test_monotile_kernel(tile_height / 2, tile_width / 2, 0, iters_per_pass);
}

TEST_CASE("monotile::StencilUpdateKernel (partial pipeline)", "[monotile::StencilUpdateKernel]") {
    static_assert(iters_per_pass != 1);
    test_monotile_kernel(tile_height, tile_width, 0, iters_per_pass - 1);
}

TEST_CASE("monotile::StencilUpdateKernel (noop)", "[monotile::StencilUpdateKernel]") {
    test_monotile_kernel(tile_height, tile_width, 0, 0);
}

TEST_CASE("monotile::StencilUpdateKernel (incomplete pipeline, offset != 0)",
          "[monotile::StencilUpdateKernel]") {
    test_monotile_kernel(tile_height, tile_width, iters_per_pass / 2, iters_per_pass);
}

template <typename TDVStrategy> void test_monotile_update() {
    using StencilUpdateImpl = StencilUpdate<FPGATransFunc<1>, n_processing_elements, tile_height,
                                            tile_width, TDVStrategy>;
    using GridImpl = Grid<Cell>;
    static_assert(concepts::StencilUpdate<StencilUpdateImpl, FPGATransFunc<1>, GridImpl>);

    for (std::size_t grid_height = tile_height / 2; grid_height < tile_height; grid_height += 1) {
        for (std::size_t grid_width = tile_width / 2; grid_width < tile_width; grid_width += 1) {
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_height, grid_width, 0,
                                                             iters_per_pass);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_height, grid_width, 1,
                                                             iters_per_pass);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_height, grid_width, 0,
                                                             iters_per_pass + 1);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_height - 1, grid_width, 0,
                                                             iters_per_pass + 1);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_height, grid_width - 1, 0,
                                                             iters_per_pass + 1);
        }
    }
}
/*
TEST_CASE("monotile::StencilUpdate", "[monotile::StencilUpdate]") {
    test_monotile_update<tdv::single_pass::InlineStrategy>();
    test_monotile_update<tdv::single_pass::PrecomputeOnDeviceStrategy>();
    test_monotile_update<tdv::single_pass::PrecomputeOnHostStrategy>();
}*/