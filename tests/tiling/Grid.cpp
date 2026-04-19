/*
 * Copyright © 2020-2026 Paderborn Center for Parallel Computing, Paderborn
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
