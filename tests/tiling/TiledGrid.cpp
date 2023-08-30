/*
 * Copyright © 2020-2023 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "../constants.hpp"
#include <CL/sycl.hpp>
#include <StencilStream/tiling/TiledGrid.hpp>
#include <catch2/catch_all.hpp>
#include <unordered_set>

using namespace stencil;
using namespace stencil::tiling;
using namespace cl::sycl;
using namespace std;

const uindex_t add_grid_width = grid_width + 1;
const uindex_t add_grid_height = grid_height + 1;

using TestGrid = TiledGrid<ID, tile_width, tile_height, halo_radius>;

// Assert that the tiled grid fulfills the grid concept.
static_assert(Grid<TestGrid, ID>);

TEST_CASE("TiledGrid::TiledGrid(uindex_t, uindex_t, uindex_t)", "[TiledGrid]") {
    TestGrid grid(add_grid_width, add_grid_height, 0);

    GenericID<uindex_t> tile_range = grid.get_tile_range();
    REQUIRE(tile_range.c == add_grid_width / tile_width + 1);
    REQUIRE(tile_range.r == add_grid_height / tile_height + 1);
}
/*
TEST_CASE("TiledGrid::TiledGrid(cl::sycl::buffer<T, 2>, T)", "[TiledGrid]") {
    buffer<ID, 2> in_buffer(range<2>(add_grid_width, add_grid_height));
    {
        auto in_buffer_ac = in_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < add_grid_width; c++) {
            for (uindex_t r = 0; r < add_grid_height; r++) {
                in_buffer_ac[c][r] = ID(c, r);
            }
        }
    }

    TestGrid grid(in_buffer);

    GenericID<uindex_t> tile_range = grid.get_tile_range();
    REQUIRE(tile_range.c == add_grid_width / tile_width + 1);
    REQUIRE(tile_range.r == add_grid_height / tile_height + 1);

    buffer<ID, 2> out_buffer(range<2>(add_grid_width, add_grid_height));
    grid.copy_to_buffer(out_buffer);

    {
        auto out_buffer_ac = out_buffer.get_access<access::mode::read>();
        for (uindex_t c = 0; c < add_grid_width; c++) {
            for (uindex_t r = 0; r < add_grid_height; r++) {
                REQUIRE(out_buffer_ac[c][r].c == c);
                REQUIRE(out_buffer_ac[c][r].r == r);
            }
        }
    }
}*/
