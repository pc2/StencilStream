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

using TestGrid = Grid<sycl::id<2>>;

// Assert that the tiled grid fulfills the grid concept.
static_assert(concepts::Grid<TestGrid, sycl::id<2>>);

TEST_CASE("tiling::Grid::Grid", "[tiling::Grid]") {
    grid_test::test_constructors<TestGrid>(add_grid_height, add_grid_width);
}

TEST_CASE("tiling::Grid::copy_from_buffer", "[tiling::Grid]") {
    grid_test::test_copy_from_buffer<TestGrid>(add_grid_height, add_grid_width);
}

TEST_CASE("tiling::Grid::copy_to_buffer", "[tiling::Grid]") {
    grid_test::test_copy_to_buffer<TestGrid>(add_grid_height, add_grid_width);
}

TEST_CASE("tiling::Grid::make_similar", "[tiling::Grid]") {
    grid_test::test_make_similar<TestGrid>(add_grid_height, add_grid_width);
}

template <std::size_t vector_length, std::size_t halo_height, std::size_t halo_width>
void test_tiling_submit_read(std::size_t grid_height, std::size_t grid_width) {
    TestGrid grid(grid_height, grid_width);
    {
        using GridAccessor = TestGrid::template GridAccessor<access::mode::read_write>;
        GridAccessor ac(grid);
        for (std::size_t r = 0; r < grid_height; r++) {
            for (std::size_t c = 0; c < grid_width; c++) {
                ac[r][c] = sycl::id<2>(r, c);
            }
        }
    }

    sycl::queue input_kernel_queue =
        sycl::queue(sycl::device(), {sycl::property::queue::in_order{}});
    sycl::queue working_queue = sycl::queue(sycl::device(), {sycl::property::queue::in_order{}});

    sycl::range<2> tile_range = grid.get_tile_range<tile_height, tile_width>();
    for (std::size_t tile_r = 0; tile_r < tile_range[0]; tile_r++) {
        for (std::size_t tile_c = 0; tile_c < tile_range[1]; tile_c++) {
            using in_pipe = sycl::pipe<class tiled_grid_submit_read_test_id,
                                       std::array<sycl::id<2>, vector_length>>;

            grid.template submit_read<in_pipe, vector_length, tile_height, tile_width, halo_height,
                                      halo_width>(input_kernel_queue, tile_r, tile_c,
                                                  sycl::id<2>(-1, -1));

            int output_tile_height = std::min(tile_height, grid_height - tile_r * tile_height);
            int input_tile_height = output_tile_height + 2 * halo_height;
            int row_offset = tile_r * tile_height - halo_height;

            int vect_output_tile_width = int_ceil_div<int>(
                std::min(tile_width, grid_width - tile_c * tile_width), vector_length);
            int output_tile_width = vect_output_tile_width * vector_length;
            int input_tile_width = output_tile_width + 2 * halo_width;
            int column_offset = tile_c * tile_width - halo_width;

            sycl::buffer<sycl::id<2>, 2> out_buffer(
                sycl::range<2>(input_tile_height, input_tile_width));
            working_queue.submit([&](sycl::handler &cgh) {
                accessor out_ac(out_buffer, cgh, write_only);

                cgh.single_task([=]() {
                    for (int local_r = 0; local_r < input_tile_height; local_r++) {
                        for (int local_c = 0; local_c < input_tile_width;
                             local_c += vector_length) {
                            std::array<sycl::id<2>, vector_length> read_vector = in_pipe::read();
#pragma unroll
                            for (int i_cell = 0; i_cell < vector_length; i_cell++) {
                                out_ac[local_r][local_c + i_cell] = read_vector[i_cell];
                            }
                        }
                    }
                });
            });

            sycl::host_accessor out_ac(out_buffer, read_only);
            for (int local_r = 0; local_r < input_tile_height; local_r++) {
                for (int local_c = 0; local_c < input_tile_width; local_c++) {
                    int r = local_r + row_offset;
                    int c = local_c + column_offset;
                    sycl::id<2> cell = out_ac[local_r][local_c];
                    if (r >= 0 && c >= 0 && r < grid_height && c < grid_width) {
                        REQUIRE(cell[0] == r);
                        REQUIRE(cell[1] == c);
                    } else {
                        REQUIRE(cell[0] == -1);
                        REQUIRE(cell[1] == -1);
                    }
                }
            }
        }
    }
}

TEST_CASE("tiling::Grid::submit_read", "[tiling::Grid]") {
    test_tiling_submit_read<1, 1, 1>(3 * tile_height, 3 * tile_width);
    test_tiling_submit_read<1, 1, 1>(tile_height - 1, tile_width - 1);

    test_tiling_submit_read<2, 1, 2>(3 * tile_height, 3 * tile_width);
    test_tiling_submit_read<2, 1, 2>(tile_height - 1, tile_width - 1);

    test_tiling_submit_read<4, 1, 4>(3 * tile_height, 3 * tile_width);
    test_tiling_submit_read<4, 1, 4>(tile_height - 1, tile_width - 1);
}

template <std::size_t vector_length> void test_tiling_submit_write() {
    using out_pipe =
        sycl::pipe<class tiled_grid_submit_write_test_id, std::array<sycl::id<2>, vector_length>>;

    sycl::queue output_kernel_queue =
        sycl::queue(sycl::device(), {sycl::property::queue::in_order{}});
    sycl::queue working_queue = sycl::queue(sycl::device(), {sycl::property::queue::in_order{}});

    std::size_t grid_height = 3 * tile_height - 1;
    std::size_t grid_width = 3 * tile_width - 1;

    TestGrid grid(grid_height, grid_width);

    for (std::size_t tile_r = 0; tile_r < 3; tile_r++) {
        for (std::size_t tile_c = 0; tile_c < 3; tile_c++) {
            working_queue.submit([&](sycl::handler &cgh) {
                std::size_t r_start = tile_r * tile_height;
                std::size_t r_end = r_start + std::min(tile_height, grid_height - r_start);
                std::size_t vect_c_start = (tile_c * tile_width) / vector_length;
                std::size_t vect_c_end =
                    vect_c_start +
                    int_ceil_div(std::min(tile_width, grid_width - tile_c * tile_width),
                                 vector_length);

                cgh.single_task([=]() {
                    for (std::size_t r = r_start; r < r_end; r++) {
                        for (std::size_t vect_c = vect_c_start; vect_c < vect_c_end; vect_c++) {
                            std::array<sycl::id<2>, vector_length> out_vector;
#pragma unroll
                            for (std::size_t cell_i = 0; cell_i < vector_length; cell_i++) {
                                out_vector[cell_i] =
                                    sycl::id<2>(r, vect_c * vector_length + cell_i);
                            }
                            out_pipe::write(out_vector);
                        }
                    }
                });
            });

            grid.template submit_write<out_pipe, vector_length, tile_height, tile_width>(
                output_kernel_queue, tile_r, tile_c);
        }
    }

    using GridAccessor = TestGrid::template GridAccessor<access::mode::read_write>;
    GridAccessor out_ac(grid);
    for (std::size_t r = 0; r < grid_height; r++) {
        for (std::size_t c = 0; c < grid_width; c++) {
            REQUIRE(out_ac[r][c][0] == r);
            REQUIRE(out_ac[r][c][1] == c);
        }
    }
}

TEST_CASE("tiling::Grid::submit_write", "[tiling::Grid]") {
    test_tiling_submit_write<1>();
    test_tiling_submit_write<2>();
    test_tiling_submit_write<4>();
}