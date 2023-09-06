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
#include "../GridTest.hpp"
#include "../constants.hpp"
#include <StencilStream/monotile/Grid.hpp>

using namespace stencil;
using namespace stencil::monotile;
using namespace cl::sycl;
using namespace std;

using TestGrid = Grid<ID, tile_width, tile_height, 64>;

// Assert that the monotile grid fulfills the grid concept.
static_assert(concepts::Grid<TestGrid, ID>);

TEST_CASE("monotile::Grid::Grid", "[monotile::Grid]") {
    grid_test::test_constructors<TestGrid>(tile_width, tile_height);
}

TEST_CASE("monotile::Grid::copy_{from|to}_buffer", "[monotile::Grid]") {
    grid_test::test_copy_from_to_buffer<TestGrid>(tile_width, tile_height);
}

TEST_CASE("monotile::Grid::make_similar", "[monotile::Grid]") {
    grid_test::test_make_similar<TestGrid>(tile_width, tile_height);
}

TEST_CASE("monotile::Grid::submit_read", "[monotile::Grid]") {
    buffer<ID, 2> in_buffer = range<2>(tile_width, tile_height);
    buffer<ID, 2> out_buffer = range<2>(tile_width, tile_height);
    {
        auto in_buffer_ac = in_buffer.get_access<cl::sycl::access::mode::discard_write>();
        for (stencil::index_t c = 0; c < tile_width; c++) {
            for (stencil::index_t r = 0; r < tile_height; r++) {
                in_buffer_ac[c][r] = stencil::ID(c, r);
            }
        }
    }

    cl::sycl::queue queue;

    TestGrid grid = in_buffer;
    using in_pipe = cl::sycl::pipe<class monotile_grid_submit_read_test_id, ID>;
    grid.template submit_read<in_pipe>(queue);

    queue.submit([&](handler &cgh) {
        auto out_ac = out_buffer.get_access<access::mode::discard_write>(cgh);

        cgh.single_task<class monotile_grid_submit_read_test_kernel>([=]() {
            for (index_t c = 0; c < tile_width; c++) {
                for (index_t r = 0; r < tile_height; r++) {
                    out_ac[c][r] = in_pipe::read();
                }
            }
        });
    });

    auto out_buffer_ac = out_buffer.get_access<access::mode::read>();
    for (index_t c = 0; c < tile_width; c++) {
        for (index_t r = 0; r < tile_height; r++) {
            REQUIRE(out_buffer_ac[c][r] == ID(c, r));
        }
    }
}

TEST_CASE("monotile::Grid::submit_write", "[monotile::Grid]") {
    using out_pipe = cl::sycl::pipe<class monotile_grid_submit_write_test_id, ID>;

    cl::sycl::queue queue;
    queue.submit([&](cl::sycl::handler &cgh) {
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

    cl::sycl::buffer<ID, 2> out_buffer = cl::sycl::range<2>(tile_width, tile_height);
    grid.copy_to_buffer(out_buffer);

    auto out_ac = out_buffer.get_access<cl::sycl::access::mode::read>();
    for (uindex_t c = 0; c < tile_width; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            REQUIRE(out_ac[c][r] == ID(c, r));
        }
    }
}