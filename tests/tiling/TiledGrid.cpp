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
    TestGrid grid(add_grid_width, add_grid_height, 42);

    GenericID<uindex_t> tile_range = grid.get_tile_range();
    REQUIRE(tile_range.c == add_grid_width / tile_width + 1);
    REQUIRE(tile_range.r == add_grid_height / tile_height + 1);
    REQUIRE(grid.get_i_generation() == 42);
}

TEST_CASE("TiledGrid::copy_{from|to}_buffer(cl::sycl::buffer<Cell, 2>)", "[TiledGrid]") {
    buffer<ID, 2> in_buffer(range<2>(add_grid_width, add_grid_height));
    buffer<ID, 2> out_buffer(range<2>(add_grid_width, add_grid_height));
    {
        auto in_buffer_ac = in_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < add_grid_width; c++) {
            for (uindex_t r = 0; r < add_grid_height; r++) {
                in_buffer_ac[c][r] = ID(c, r);
            }
        }
    }

    TestGrid grid(add_grid_width, add_grid_height, 0);
    grid.copy_from_buffer(in_buffer);
    grid.copy_to_buffer(out_buffer);

    {
        auto out_buffer_ac = out_buffer.get_access<access::mode::read>();
        for (uindex_t c = 0; c < add_grid_width; c++) {
            for (uindex_t r = 0; r < add_grid_height; r++) {
                REQUIRE(out_buffer_ac[c][r] == ID(c, r));
            }
        }
    }
}

TEST_CASE("TiledGrid::make_similar()", "[TiledGrid]") {
    TestGrid grid(add_grid_width, add_grid_height, 42);
    TestGrid similar_grid = grid.make_similar();
    REQUIRE(similar_grid.get_grid_width() == add_grid_width);
    REQUIRE(similar_grid.get_grid_height() == add_grid_height);
    REQUIRE(similar_grid.get_i_generation() == 42);
}

TEST_CASE("TiledGrid::submit_read(cl::sycl::queue, index_t, index_t)", "[TiledGrid]") {
    buffer<ID, 2> in_buffer(range<2>(3 * tile_width, 3 * tile_height));
    {
        auto in_buffer_ac = in_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < 3 * tile_width; c++) {
            for (uindex_t r = 0; r < 3 * tile_height; r++) {
                in_buffer_ac[c][r] = ID(c, r);
            }
        }
    }
    TestGrid grid(in_buffer);

    cl::sycl::queue queue;

    using in_pipe = cl::sycl::pipe<class tiled_grid_submit_read_test_id, ID>;
    grid.template submit_read<in_pipe>(queue, 1, 1);

    cl::sycl::buffer<bool, 1> result_buffer = cl::sycl::range<1>(1);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto result_ac = result_buffer.get_access<cl::sycl::access::mode::discard_write>(cgh);
        uindex_t c_start = tile_width - halo_radius;
        uindex_t c_end = 2 * tile_width + halo_radius;
        uindex_t r_start = tile_height - halo_radius;
        uindex_t r_end = 2 * tile_height + halo_radius;

        cgh.single_task([=]() {
            bool correct_input = true;
            for (uindex_t c = c_start; c < c_end; c++) {
                for (uindex_t r = r_start; r < r_end; r++) {
                    ID read_value = in_pipe::read();
                    correct_input &= read_value == ID(c, r);
                }
            }
            result_ac[0] = correct_input;
        });
    });

    auto result_ac = result_buffer.get_access<cl::sycl::access::mode::read>();
    REQUIRE(result_ac[0]);
}

TEST_CASE("TiledGrid::submit_write(cl::sycl::queue, index_t index_t)", "[TiledGrid]") {
    using out_pipe = cl::sycl::pipe<class tiled_grid_submit_write_test_id, ID>;

    cl::sycl::queue queue;
    queue.submit([&](cl::sycl::handler &cgh) {
        cgh.single_task([=]() {
            for (uindex_t c = tile_width; c < 2 * tile_width; c++) {
                for (uindex_t r = tile_height; r < 2 * tile_height; r++) {
                    out_pipe::write(ID(c, r));
                }
            }
        });
    });

    TestGrid grid(3 * tile_width, 3 * tile_height, 0);
    grid.template submit_write<out_pipe>(queue, 1, 1);

    cl::sycl::buffer<ID, 2> out_buffer = cl::sycl::range<2>(3 * tile_width, 3 * tile_height);
    grid.copy_to_buffer(out_buffer);

    auto out_ac = out_buffer.get_access<cl::sycl::access::mode::read>();
    for (uindex_t c = tile_width; c < 2 * tile_width; c++) {
        for (uindex_t r = tile_height; r < 2 * tile_height; r++) {
            REQUIRE(out_ac[c][r] == ID(c, r));
        }
    }
}
