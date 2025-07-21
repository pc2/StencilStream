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
#include <CL/sycl.hpp>
#include <StencilStream/tiling/Grid.hpp>
#include <catch2/catch_all.hpp>
#include <sycl/sycl.hpp>
#include <unordered_set>

using namespace stencil;
using namespace stencil::tiling;
using namespace sycl;
using namespace std;

// Assert that the tiled grid fulfills the grid concept.
static_assert(concepts::Grid<Grid<sycl::id<2>, 1>, sycl::id<2>>);

// Using an arbitrary, but non-quadratic and non-power-of-two grid size for testing.
// This should catch most of the nasties.
TEST_CASE("tiling::Grid::Grid", "[tiling::Grid]") {
    grid_test::test_constructors<Grid<sycl::id<2>, 1>>(129, 65);
}

TEST_CASE("tiling::Grid::copy_from_buffer", "[tiling::Grid]") {
    grid_test::test_copy_from_buffer<Grid<sycl::id<2>, 1>>(129, 65);
}

TEST_CASE("tiling::Grid::copy_to_buffer", "[tiling::Grid]") {
    grid_test::test_copy_to_buffer<Grid<sycl::id<2>, 1>>(129, 65);
}

TEST_CASE("tiling::Grid::make_similar", "[tiling::Grid]") {
    grid_test::test_make_similar<Grid<sycl::id<2>, 1>>(129, 65);
}

TEST_CASE("tiling::Grid::get_grid_range", "[tiling::Grid]") {
    Grid<sycl::id<2>, 1> grid_1(64, 64);
    REQUIRE(grid_1.get_grid_range(false) == sycl::range<2>(64, 64));
    REQUIRE(grid_1.get_grid_range(true) == sycl::range<2>(64, 64));
    grid_1 = Grid<sycl::id<2>, 1>(63, 63);
    REQUIRE(grid_1.get_grid_range(false) == sycl::range<2>(63, 63));
    REQUIRE(grid_1.get_grid_range(true) == sycl::range<2>(63, 63));

    Grid<sycl::id<2>, 2> grid_2(64, 64);
    REQUIRE(grid_2.get_grid_range(false) == sycl::range<2>(64, 64));
    REQUIRE(grid_2.get_grid_range(true) == sycl::range<2>(64, 32));
    grid_2 = Grid<sycl::id<2>, 2>(63, 63);
    REQUIRE(grid_2.get_grid_range(false) == sycl::range<2>(63, 63));
    REQUIRE(grid_2.get_grid_range(true) == sycl::range<2>(63, 32));
}

TEST_CASE("tiling::Grid::get_tile_id_range", "[tiling::Grid]") {
    Grid<sycl::id<2>, 2> grid(64, 64);
    REQUIRE(grid.get_tile_id_range(sycl::range<2>(64, 64)) == sycl::range<2>(1, 1));
    REQUIRE(grid.get_tile_id_range(sycl::range<2>(48, 48)) == sycl::range<2>(2, 2));
    REQUIRE(grid.get_tile_id_range(sycl::range<2>(32, 32)) == sycl::range<2>(2, 2));
    REQUIRE(grid.get_tile_id_range(sycl::range<2>(31, 30)) == sycl::range<2>(3, 3));

    // Tile width not a multiple of spatial parallelism.
    REQUIRE_THROWS_AS(grid.get_tile_id_range(sycl::range<2>(32, 31)), std::invalid_argument);
}

TEST_CASE("tiling::Grid::get_tile_offset", "[tiling::Grid]") {
    sycl::range<2> grid_range(64, 64);
    sycl::range<2> max_tile_range(48, 32);
    Grid<sycl::id<2>, 2> grid(grid_range);

    REQUIRE(grid.get_tile_offset(sycl::id<2>(0, 0), max_tile_range, false) == sycl::id<2>(0, 0));
    REQUIRE(grid.get_tile_offset(sycl::id<2>(0, 0), max_tile_range, true) == sycl::id<2>(0, 0));
    REQUIRE(grid.get_tile_offset(sycl::id<2>(0, 1), max_tile_range, false) == sycl::id<2>(0, 32));
    REQUIRE(grid.get_tile_offset(sycl::id<2>(0, 1), max_tile_range, true) == sycl::id<2>(0, 16));
    REQUIRE(grid.get_tile_offset(sycl::id<2>(1, 0), max_tile_range, false) == sycl::id<2>(48, 0));
    REQUIRE(grid.get_tile_offset(sycl::id<2>(1, 0), max_tile_range, true) == sycl::id<2>(48, 0));
    REQUIRE(grid.get_tile_offset(sycl::id<2>(1, 1), max_tile_range, false) == sycl::id<2>(48, 32));
    REQUIRE(grid.get_tile_offset(sycl::id<2>(1, 1), max_tile_range, true) == sycl::id<2>(48, 16));

    // Tile width not a multiple of spatial parallelism.
    REQUIRE_THROWS_AS(grid.get_tile_offset(sycl::id<2>(0, 0), sycl::range<2>(32, 31), false),
                      std::invalid_argument);
    // Tile row out of range.
    REQUIRE_THROWS_AS(grid.get_tile_offset(sycl::id<2>(2, 1), max_tile_range, false),
                      std::out_of_range);
    // Tile column out of range.
    REQUIRE_THROWS_AS(grid.get_tile_offset(sycl::id<2>(1, 2), max_tile_range, false),
                      std::out_of_range);
}

TEST_CASE("tiling::Grid::get_haloed_tile_offset", "[tiling::Grid]") {
    sycl::range<2> grid_range(64, 64);
    sycl::range<2> max_tile_range(48, 32);
    Grid<sycl::id<2>, 2> grid(grid_range);

    std::array<std::ptrdiff_t, 2> result;

    // Are the properties of get_tile_offset preserved if the halo range is zero?
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 0), max_tile_range, sycl::range<2>(0, 0),
                                         false, false);
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 0), max_tile_range, sycl::range<2>(0, 0),
                                         true, false);
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 1), max_tile_range, sycl::range<2>(0, 0),
                                         false, false);
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 32);
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 1), max_tile_range, sycl::range<2>(0, 0),
                                         true, false);
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 16);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 0), max_tile_range, sycl::range<2>(0, 0),
                                         false, false);
    REQUIRE(result[0] == 48);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 0), max_tile_range, sycl::range<2>(0, 0),
                                         true, false);
    REQUIRE(result[0] == 48);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 1), max_tile_range, sycl::range<2>(0, 0),
                                         false, false);
    REQUIRE(result[0] == 48);
    REQUIRE(result[1] == 32);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 1), max_tile_range, sycl::range<2>(0, 0),
                                         true, false);
    REQUIRE(result[0] == 48);
    REQUIRE(result[1] == 16);

    // Plain halo additions, including out-of-bounds cells
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 0), max_tile_range, sycl::range<2>(16, 32),
                                         false, false);
    REQUIRE(result[0] == -16);
    REQUIRE(result[1] == -32);
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 0), max_tile_range, sycl::range<2>(16, 32),
                                         true, false);
    REQUIRE(result[0] == -16);
    REQUIRE(result[1] == -16);
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 1), max_tile_range, sycl::range<2>(16, 32),
                                         false, false);
    REQUIRE(result[0] == -16);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 1), max_tile_range, sycl::range<2>(16, 32),
                                         true, false);
    REQUIRE(result[0] == -16);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 0), max_tile_range, sycl::range<2>(16, 32),
                                         false, false);
    REQUIRE(result[0] == 32);
    REQUIRE(result[1] == -32);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 0), max_tile_range, sycl::range<2>(16, 32),
                                         true, false);
    REQUIRE(result[0] == 32);
    REQUIRE(result[1] == -16);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 1), max_tile_range, sycl::range<2>(16, 32),
                                         false, false);
    REQUIRE(result[0] == 32);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 1), max_tile_range, sycl::range<2>(16, 32),
                                         true, false);
    REQUIRE(result[0] == 32);
    REQUIRE(result[1] == 0);

    // Halo additions, including only valid cells
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 0), max_tile_range, sycl::range<2>(16, 32),
                                         false, true);
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 0), max_tile_range, sycl::range<2>(16, 32),
                                         true, true);
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 1), max_tile_range, sycl::range<2>(16, 32),
                                         false, true);
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(0, 1), max_tile_range, sycl::range<2>(16, 32),
                                         true, true);
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 0), max_tile_range, sycl::range<2>(16, 32),
                                         false, true);
    REQUIRE(result[0] == 32);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 0), max_tile_range, sycl::range<2>(16, 32),
                                         true, true);
    REQUIRE(result[0] == 32);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 1), max_tile_range, sycl::range<2>(16, 32),
                                         false, true);
    REQUIRE(result[0] == 32);
    REQUIRE(result[1] == 0);
    result = grid.get_haloed_tile_offset(sycl::id<2>(1, 1), max_tile_range, sycl::range<2>(16, 32),
                                         true, true);
    REQUIRE(result[0] == 32);
    REQUIRE(result[1] == 0);

    // Exceptions
    // Tile width not a multiple of spatial parallelism.
    REQUIRE_THROWS_AS(grid.get_haloed_tile_offset(sycl::id<2>(0, 0), sycl::range<2>(32, 31),
                                                  sycl::range<2>(16, 32), false, false),
                      std::invalid_argument);
    // Halo width not a multiple of spatial parallelism.
    REQUIRE_THROWS_AS(grid.get_haloed_tile_offset(sycl::id<2>(0, 0), sycl::range<2>(32, 32),
                                                  sycl::range<2>(16, 31), false, false),
                      std::invalid_argument);
    // Tile row out of bounds.
    REQUIRE_THROWS_AS(grid.get_haloed_tile_offset(sycl::id<2>(2, 1), sycl::range<2>(32, 32),
                                                  sycl::range<2>(16, 32), false, false),
                      std::out_of_range);
    // Tile column out of bounds.
    REQUIRE_THROWS_AS(grid.get_haloed_tile_offset(sycl::id<2>(1, 2), sycl::range<2>(32, 32),
                                                  sycl::range<2>(16, 32), false, false),
                      std::out_of_range);
}

TEST_CASE("tiling::Grid::get_tile_range", "[tiling::Grid]") {
    sycl::range<2> grid_range(64, 64);
    sycl::range<2> max_tile_range(48, 48);
    Grid<sycl::id<2>, 2> grid(grid_range);

    REQUIRE(grid.get_tile_range(sycl::id<2>(0, 0), max_tile_range, false) ==
            sycl::range<2>(48, 48));
    REQUIRE(grid.get_tile_range(sycl::id<2>(0, 0), max_tile_range, true) == sycl::range<2>(48, 24));
    REQUIRE(grid.get_tile_range(sycl::id<2>(0, 1), max_tile_range, false) ==
            sycl::range<2>(48, 16));
    REQUIRE(grid.get_tile_range(sycl::id<2>(0, 1), max_tile_range, true) == sycl::range<2>(48, 8));
    REQUIRE(grid.get_tile_range(sycl::id<2>(1, 0), max_tile_range, false) ==
            sycl::range<2>(16, 48));
    REQUIRE(grid.get_tile_range(sycl::id<2>(1, 0), max_tile_range, true) == sycl::range<2>(16, 24));
    REQUIRE(grid.get_tile_range(sycl::id<2>(1, 1), max_tile_range, false) ==
            sycl::range<2>(16, 16));
    REQUIRE(grid.get_tile_range(sycl::id<2>(1, 1), max_tile_range, true) == sycl::range<2>(16, 8));

    // Tile width not a multiple of spatial parallelism.
    REQUIRE_THROWS_AS(grid.get_tile_range(sycl::id<2>(0, 0), sycl::range<2>(32, 31), false),
                      std::invalid_argument);
    // Tile row out of range.
    REQUIRE_THROWS_AS(grid.get_tile_range(sycl::id<2>(2, 1), max_tile_range, false),
                      std::out_of_range);
    // Tile column out of range.
    REQUIRE_THROWS_AS(grid.get_tile_range(sycl::id<2>(1, 2), max_tile_range, false),
                      std::out_of_range);
}

TEST_CASE("tiling::Grid::get_haloed_tile_range", "[tiling::Grid]") {
    sycl::range<2> grid_range(64, 64);
    sycl::range<2> max_tile_range(48, 48);
    sycl::range<2> halo_range(16, 32);
    Grid<sycl::id<2>, 2> grid(grid_range);

    // Are the properties of get_tile_range preserved if the halo range is zero?
    sycl::range<2> result = grid.get_haloed_tile_range(sycl::id<2>(0, 0), max_tile_range,
                                                       sycl::range<2>(0, 0), false, false);
    REQUIRE(result == sycl::range<2>(48, 48));
    result = grid.get_haloed_tile_range(sycl::id<2>(0, 0), max_tile_range, sycl::range<2>(0, 0),
                                        true, false);
    REQUIRE(result == sycl::range<2>(48, 24));
    result = grid.get_haloed_tile_range(sycl::id<2>(0, 1), max_tile_range, sycl::range<2>(0, 0),
                                        false, false);
    REQUIRE(result == sycl::range<2>(48, 16));
    result = grid.get_haloed_tile_range(sycl::id<2>(0, 1), max_tile_range, sycl::range<2>(0, 0),
                                        true, false);
    REQUIRE(result == sycl::range<2>(48, 8));
    result = grid.get_haloed_tile_range(sycl::id<2>(1, 0), max_tile_range, sycl::range<2>(0, 0),
                                        false, false);
    REQUIRE(result == sycl::range<2>(16, 48));
    result = grid.get_haloed_tile_range(sycl::id<2>(1, 0), max_tile_range, sycl::range<2>(0, 0),
                                        true, false);
    REQUIRE(result == sycl::range<2>(16, 24));
    result = grid.get_haloed_tile_range(sycl::id<2>(1, 1), max_tile_range, sycl::range<2>(0, 0),
                                        false, false);
    REQUIRE(result == sycl::range<2>(16, 16));
    result = grid.get_haloed_tile_range(sycl::id<2>(1, 1), max_tile_range, sycl::range<2>(0, 0),
                                        true, false);
    REQUIRE(result == sycl::range<2>(16, 8));

    // Simple halo addition, including invalid cells
    result =
        grid.get_haloed_tile_range(sycl::id<2>(0, 0), max_tile_range, halo_range, false, false);
    REQUIRE(result == sycl::range<2>(80, 112));
    result = grid.get_haloed_tile_range(sycl::id<2>(0, 0), max_tile_range, halo_range, true, false);
    REQUIRE(result == sycl::range<2>(80, 56));
    result =
        grid.get_haloed_tile_range(sycl::id<2>(0, 1), max_tile_range, halo_range, false, false);
    REQUIRE(result == sycl::range<2>(80, 80));
    result = grid.get_haloed_tile_range(sycl::id<2>(0, 1), max_tile_range, halo_range, true, false);
    REQUIRE(result == sycl::range<2>(80, 40));
    result =
        grid.get_haloed_tile_range(sycl::id<2>(1, 0), max_tile_range, halo_range, false, false);
    REQUIRE(result == sycl::range<2>(48, 112));
    result = grid.get_haloed_tile_range(sycl::id<2>(1, 0), max_tile_range, halo_range, true, false);
    REQUIRE(result == sycl::range<2>(48, 56));
    result =
        grid.get_haloed_tile_range(sycl::id<2>(1, 1), max_tile_range, halo_range, false, false);
    REQUIRE(result == sycl::range<2>(48, 80));
    result = grid.get_haloed_tile_range(sycl::id<2>(1, 1), max_tile_range, halo_range, true, false);
    REQUIRE(result == sycl::range<2>(48, 40));

    // Including halos, but only with valid cells.
    result = grid.get_haloed_tile_range(sycl::id<2>(0, 0), max_tile_range, halo_range, false, true);
    REQUIRE(result == sycl::range<2>(64, 64));
    result = grid.get_haloed_tile_range(sycl::id<2>(0, 0), max_tile_range, halo_range, true, true);
    REQUIRE(result == sycl::range<2>(64, 32));
    result = grid.get_haloed_tile_range(sycl::id<2>(0, 1), max_tile_range, halo_range, false, true);
    REQUIRE(result == sycl::range<2>(64, 48));
    result = grid.get_haloed_tile_range(sycl::id<2>(0, 1), max_tile_range, halo_range, true, true);
    REQUIRE(result == sycl::range<2>(64, 24));
    result = grid.get_haloed_tile_range(sycl::id<2>(1, 0), max_tile_range, halo_range, false, true);
    REQUIRE(result == sycl::range<2>(32, 64));
    result = grid.get_haloed_tile_range(sycl::id<2>(1, 0), max_tile_range, halo_range, true, true);
    REQUIRE(result == sycl::range<2>(32, 32));
    result = grid.get_haloed_tile_range(sycl::id<2>(1, 1), max_tile_range, halo_range, false, true);
    REQUIRE(result == sycl::range<2>(32, 48));
    result = grid.get_haloed_tile_range(sycl::id<2>(1, 1), max_tile_range, halo_range, true, true);
    REQUIRE(result == sycl::range<2>(32, 24));

    // Tile width not a multiple of spatial parallelism.
    REQUIRE_THROWS_AS(grid.get_haloed_tile_range(sycl::id<2>(0, 0), sycl::range<2>(32, 31),
                                                 halo_range, false, false),
                      std::invalid_argument);
    // Halo width not a multiple of spatial parallelism.
    REQUIRE_THROWS_AS(grid.get_haloed_tile_range(sycl::id<2>(0, 0), max_tile_range,
                                                 sycl::range<2>(16, 31), false, false),
                      std::invalid_argument);
    // Tile row out of range.
    REQUIRE_THROWS_AS(
        grid.get_haloed_tile_range(sycl::id<2>(2, 1), max_tile_range, halo_range, false, false),
        std::out_of_range);
    // Tile column out of range.
    REQUIRE_THROWS_AS(
        grid.get_haloed_tile_range(sycl::id<2>(1, 2), max_tile_range, halo_range, false, false),
        std::out_of_range);
}

template <std::size_t spatial_parallelism, std::size_t tile_height, std::size_t tile_width,
          std::size_t halo_height, std::size_t halo_width>
void test_tiling_submit_read(std::size_t grid_height, std::size_t grid_width) {
    using TestGrid = Grid<sycl::id<2>, spatial_parallelism>;
    using CellVector = typename TestGrid::CellVector;

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

    sycl::range<2> max_tile_range = sycl::range<2>(tile_height, tile_width);
    sycl::range<2> tile_id_range = grid.get_tile_id_range(max_tile_range);
    for (std::size_t tile_r = 0; tile_r < tile_id_range[0]; tile_r++) {
        for (std::size_t tile_c = 0; tile_c < tile_id_range[1]; tile_c++) {
            using in_pipe = sycl::pipe<class tiled_grid_submit_read_test_id, CellVector>;

            grid.template submit_read<in_pipe, tile_height, tile_width, halo_height, halo_width>(
                input_kernel_queue, tile_r, tile_c, sycl::id<2>(-1, -1));

            int output_tile_height = std::min(tile_height, grid_height - tile_r * tile_height);
            int input_tile_height = output_tile_height + 2 * halo_height;
            int row_offset = tile_r * tile_height - halo_height;

            int vect_output_tile_width = int_ceil_div<int>(
                std::min(tile_width, grid_width - tile_c * tile_width), spatial_parallelism);
            int output_tile_width = vect_output_tile_width * spatial_parallelism;
            int input_tile_width = output_tile_width + 2 * halo_width;
            int column_offset = tile_c * tile_width - halo_width;

            sycl::buffer<sycl::id<2>, 2> out_buffer(
                sycl::range<2>(input_tile_height, input_tile_width));
            working_queue.submit([&](sycl::handler &cgh) {
                accessor out_ac(out_buffer, cgh, write_only);

                cgh.single_task([=]() {
                    for (int local_r = 0; local_r < input_tile_height; local_r++) {
                        for (int local_c = 0; local_c < input_tile_width;
                             local_c += spatial_parallelism) {
                            CellVector read_vector = in_pipe::read();
#pragma unroll
                            for (int i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                                out_ac[local_r][local_c + i_cell] = read_vector.value[i_cell];
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
    constexpr std::size_t tile_height = 64;
    constexpr std::size_t tile_width = 32;

    test_tiling_submit_read<1, tile_height, tile_width, 1, 1>(3 * tile_height, 3 * tile_width);
    test_tiling_submit_read<1, tile_height, tile_width, 1, 1>(tile_height - 1, tile_width - 1);

    test_tiling_submit_read<2, tile_height, tile_width, 1, 2>(3 * tile_height, 3 * tile_width);
    test_tiling_submit_read<2, tile_height, tile_width, 1, 2>(tile_height - 1, tile_width - 1);

    test_tiling_submit_read<4, tile_height, tile_width, 1, 4>(3 * tile_height, 3 * tile_width);
    test_tiling_submit_read<4, tile_height, tile_width, 1, 4>(tile_height - 1, tile_width - 1);
}
