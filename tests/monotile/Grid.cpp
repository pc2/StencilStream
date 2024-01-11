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
#include <StencilStream/monotile/Grid.hpp>

using namespace stencil;
using namespace stencil::monotile;
using namespace sycl;
using namespace std;

using TestGrid = Grid<ID, 64>;

// Assert that the monotile grid fulfills the grid concept.
static_assert(concepts::Grid<TestGrid, ID>);

TEST_CASE("monotile::Grid::Grid", "[monotile::Grid]") {
    grid_test::test_constructors<TestGrid>(tile_width, tile_height);
}

TEST_CASE("monotile::Grid::copy_from_buffer", "[monotile::Grid]") {
    grid_test::test_copy_from_buffer<TestGrid>(tile_width, tile_height);
}

TEST_CASE("monotile::Grid::copy_to_buffer", "[monotile::Grid]") {
    grid_test::test_copy_to_buffer<TestGrid>(tile_width, tile_height);
}

TEST_CASE("monotile::Grid::make_similar", "[monotile::Grid]") {
    grid_test::test_make_similar<TestGrid>(tile_width, tile_height);
}

TEST_CASE("monotile::Grid::submit_read", "[monotile::Grid]") {
    TestGrid in_grid(tile_width, tile_height);
    {
        TestGrid::GridAccessor<access::mode::read_write> in_grid_ac(in_grid);
        for (stencil::uindex_t c = 0; c < tile_width; c++) {
            for (stencil::uindex_t r = 0; r < tile_height; r++) {
                in_grid_ac[c][r] = ID(c, r);
            }
        }
    }

    sycl::queue queue;

    using in_pipe = sycl::pipe<class monotile_grid_submit_read_test_id, ID>;
    in_grid.template submit_read<in_pipe>(queue);

    buffer<ID, 2> out_buffer = range<2>(tile_width, tile_height);
    queue.submit([&](handler &cgh) {
        accessor out_ac(out_buffer, cgh, sycl::write_only);

        cgh.single_task<class monotile_grid_submit_read_test_kernel>([=]() {
            for (index_t c = 0; c < tile_width; c++) {
                for (index_t r = 0; r < tile_height; r++) {
                    out_ac[c][r] = in_pipe::read();
                }
            }
        });
    });

    host_accessor out_buffer_ac(out_buffer, sycl::read_only);
    for (index_t c = 0; c < tile_width; c++) {
        for (index_t r = 0; r < tile_height; r++) {
            REQUIRE(out_buffer_ac[c][r] == ID(c, r));
        }
    }
}

TEST_CASE("monotile::Grid::submit_write", "[monotile::Grid]") {
    using out_pipe = sycl::pipe<class monotile_grid_submit_write_test_id, ID>;

    sycl::queue queue;
    queue.submit([&](sycl::handler &cgh) {
        cgh.single_task<class monotile_grid_submit_write_test_kernel>([=]() {
            for (uindex_t c = 0; c < tile_width; c++) {
                for (uindex_t r = 0; r < tile_height; r++) {
                    out_pipe::write(ID(c, r));
                }
            }
        });
    });

    TestGrid grid(tile_width, tile_height);
    grid.template submit_write<out_pipe>(queue);

    TestGrid::GridAccessor<access::mode::read_write> out_ac(grid);
    for (uindex_t c = 0; c < tile_width; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            REQUIRE(out_ac[c][r] == ID(c, r));
        }
    }
}