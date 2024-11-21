/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
 * University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the “Software”), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "../GridTest.hpp"
#include "../constants.hpp"
#include <CL/sycl.hpp>
#include <StencilStream/tiling/Grid.hpp>
#include <catch2/catch_all.hpp>
#include <unordered_set>

using namespace stencil;
using namespace stencil::tiling;
using namespace sycl;
using namespace std;

const std::size_t add_grid_height = grid_height + 1;
const std::size_t add_grid_width = grid_width + 1;

using TestGrid = Grid<sycl::id<2>, tile_height, tile_width, halo_radius>;

// Assert that the tiled grid fulfills the grid concept.
static_assert(concepts::Grid<TestGrid, sycl::id<2>>);

TEST_CASE("tiling::Grid::Grid", "[tiling::Grid]") { grid_test::test_constructors<TestGrid>(add_grid_height, add_grid_width); }

TEST_CASE("tiling::Grid::copy_from_buffer", "[tiling::Grid]") { grid_test::test_copy_from_buffer<TestGrid>(add_grid_height, add_grid_width); }

TEST_CASE("tiling::Grid::copy_to_buffer", "[tiling::Grid]") { grid_test::test_copy_to_buffer<TestGrid>(add_grid_height, add_grid_width); }

TEST_CASE("tiling::Grid::make_similar", "[tiling::Grid]") { grid_test::test_make_similar<TestGrid>(add_grid_height, add_grid_width); }

TEST_CASE("tiling::Grid::submit_read", "[tiling::Grid]") {
    TestGrid grid(3 * tile_height, 3 * tile_width);
    {
        using GridAccessor = TestGrid::template GridAccessor<access::mode::read_write>;
        GridAccessor ac(grid);
        for (std::size_t r = 0; r < 3 * tile_height; r++) {
            for (std::size_t c = 0; c < 3 * tile_width; c++) {
                ac[r][c] = sycl::id<2>(r, c);
            }
        }
    }

    sycl::queue input_kernel_queue = sycl::queue(sycl::device(), {sycl::property::queue::in_order{}});
    sycl::queue working_queue = sycl::queue(sycl::device(), {sycl::property::queue::in_order{}});

    for (std::size_t tile_r = 0; tile_r < 3; tile_r++) {
        for (std::size_t tile_c = 0; tile_c < 3; tile_c++) {
            using in_pipe = sycl::pipe<class tiled_grid_submit_read_test_id, sycl::id<2>>;
            grid.template submit_read<in_pipe>(input_kernel_queue, tile_r, tile_c, sycl::id<2>(-1, -1));

            sycl::buffer<bool, 1> result_buffer = sycl::range<1>(1);
            working_queue.submit([&](sycl::handler &cgh) {
                accessor result_ac(result_buffer, cgh, write_only);
                std::size_t r_start = tile_r * tile_height;
                std::size_t r_end = (tile_r + 1) * tile_height + 2 * halo_radius;
                std::size_t c_start = tile_c * tile_width;
                std::size_t c_end = (tile_c + 1) * tile_width + 2 * halo_radius;

                cgh.single_task<class tiled_grid_submit_read_test_kernel>([=]() {
                    bool correct_input = true;
                    for (std::size_t r = r_start; r < r_end; r++) {
                        for (std::size_t c = c_start; c < c_end; c++) {
                            sycl::id<2> read_value = in_pipe::read();
                            if (r >= halo_radius && c >= halo_radius && r < 3 * tile_height + halo_radius && c < 3 * tile_width + halo_radius) {
                                correct_input &= read_value == sycl::id<2>(r - halo_radius, c - halo_radius);
                            } else {
                                correct_input &= read_value == sycl::id<2>(-1, -1);
                            }
                        }
                    }
                    result_ac[0] = correct_input;
                });
            });

            CHECK(host_accessor(result_buffer)[0]);
        }
    }
}

TEST_CASE("tiling::Grid::submit_write", "[tiling::Grid]") {
    using out_pipe = sycl::pipe<class tiled_grid_submit_write_test_id, sycl::id<2>>;

    sycl::queue output_kernel_queue = sycl::queue(sycl::device(), {sycl::property::queue::in_order{}});
    sycl::queue working_queue = sycl::queue(sycl::device(), {sycl::property::queue::in_order{}});

    TestGrid grid(3 * tile_height, 3 * tile_width);

    for (std::size_t tile_r = 0; tile_r < 3; tile_r++) {
        for (std::size_t tile_c = 0; tile_c < 3; tile_c++) {
            working_queue.submit([&](sycl::handler &cgh) {
                std::size_t r_start = tile_r * tile_height;
                std::size_t r_end = (tile_r + 1) * tile_height;
                std::size_t c_start = tile_c * tile_width;
                std::size_t c_end = (tile_c + 1) * tile_width;

                cgh.single_task<class tiled_grid_submit_write_test_kernel>([=]() {
                    for (std::size_t r = r_start; r < r_end; r++) {
                        for (std::size_t c = c_start; c < c_end; c++) {
                            out_pipe::write(sycl::id<2>(r, c));
                        }
                    }
                });
            });

            grid.template submit_write<out_pipe>(output_kernel_queue, tile_r, tile_c);
        }
    }

    using GridAccessor = TestGrid::template GridAccessor<access::mode::read_write>;
    GridAccessor out_ac(grid);
    for (std::size_t r = 0; r < 3 * tile_height; r++) {
        for (std::size_t c = 0; c < 3 * tile_width; c++) {
            REQUIRE(out_ac[r][c][0] == r);
            REQUIRE(out_ac[r][c][1] == c);
        }
    }
}
