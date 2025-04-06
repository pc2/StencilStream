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
#include "../TransFuncs.hpp"
#include <StencilStream/monotile/StencilUpdateKernel.hpp>
#include <catch2/catch_all.hpp>

using namespace sycl;
using namespace stencil;
using namespace stencil::monotile;

constexpr std::size_t stencil_radius = 2;
constexpr std::size_t tile_height = 64;
constexpr std::size_t tile_width = 32;
constexpr std::size_t temporal_parallelism = 4;
using TransFunc = HostTransFunc<stencil_radius>;

template <std::size_t spatial_parallelism>
void test_monotile_kernel(std::size_t grid_height, std::size_t grid_width,
                          std::size_t iteration_offset, std::size_t target_i_iteration) {
    using in_pipe =
        HostPipe<class MonotileExecutionKernelInPipeID, std::array<Cell, spatial_parallelism>>;
    using out_pipe =
        HostPipe<class MonotileExecutionKernelOutPipeID, std::array<Cell, spatial_parallelism>>;
    using GlobalState = tdv::single_pass::InlineStrategy::GlobalState<TransFunc, 1>;
    using KernelArgument = typename GlobalState::KernelArgument;
    using TestExecutionKernel =
        StencilUpdateKernel<TransFunc, KernelArgument, temporal_parallelism, spatial_parallelism,
                            tile_height, tile_width, in_pipe, out_pipe>;

    sycl::queue working_queue;

    for (std::size_t r = 0; r < grid_height; r++) {
        for (std::size_t c = 0; c < grid_width; c += spatial_parallelism) {
            std::array<Cell, spatial_parallelism> vector;
            for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
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
        for (std::size_t c = 0; c < grid_width; c += spatial_parallelism) {
            std::array<Cell, spatial_parallelism> vector = out_pipe::read();
            for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                if (c + i_cell >= grid_width) {
                    break;
                }
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
    std::size_t n_processing_elements = temporal_parallelism * TransFunc::n_subiterations;
    std::size_t n_steps =
        (std::log2(tile_height) - 1) * (std::log2(tile_width) - 1) * (n_processing_elements - 1);
    double progress = 0.0;
    double progress_per_step = 80 / double(n_steps);
    std::cout << "Progress monotile::StencilUpdateKernel: ";

    for (std::size_t grid_height = stencil_radius; grid_height <= tile_height; grid_height *= 2) {
        for (std::size_t grid_width = stencil_radius; grid_width <= tile_width; grid_width *= 2) {
            for (std::size_t iters = 1; iters <= temporal_parallelism; iters++) {
                test_monotile_kernel<1>(grid_height, grid_width, 0, iters);
                test_monotile_kernel<1>(grid_height, grid_width, temporal_parallelism,
                                        temporal_parallelism + iters);

                test_monotile_kernel<4>(grid_height, grid_width, 0, iters);
                test_monotile_kernel<4>(grid_height, grid_width, temporal_parallelism,
                                        temporal_parallelism + iters);

                if (grid_height > stencil_radius) {
                    test_monotile_kernel<4>(grid_height - 1, grid_width, 0, iters);
                    test_monotile_kernel<4>(grid_height - 1, grid_width, temporal_parallelism,
                                            temporal_parallelism + iters);
                }

                if (grid_width > stencil_radius) {
                    test_monotile_kernel<4>(grid_height, grid_width - 1, 0, iters);
                    test_monotile_kernel<4>(grid_height, grid_width - 1, temporal_parallelism,
                                            temporal_parallelism + iters);
                }

                if (grid_height > stencil_radius && grid_width > stencil_radius) {
                    test_monotile_kernel<4>(grid_height - 1, grid_width - 1, 0, iters);
                    test_monotile_kernel<4>(grid_height - 1, grid_width - 1, temporal_parallelism,
                                            temporal_parallelism + iters);
                }

                if (std::ceil(progress) < std::ceil(progress + progress_per_step)) {
                    std::cout << "#";
                    std::flush(std::cout);
                }
                progress += progress_per_step;
            }
        }
    }
    std::cout << std::endl;
}
