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
#include <CL/sycl.hpp>
#include <StencilStream/tiling/Grid.hpp>
#include <catch2/catch_all.hpp>
#include <unordered_set>

using namespace stencil;
using namespace stencil::tiling;
using namespace sycl;
using namespace std;

const uindex_t add_grid_width = grid_width + 1;
const uindex_t add_grid_height = grid_height + 1;

constexpr TileParameters tile_params{
    .width = tile_width,
    .height = tile_height,
    .halo_radius = halo_radius,
    .word_size = 64,
};
using TestGrid = Grid<ID, tile_params>;

// Assert that the tiled grid fulfills the grid concept.
static_assert(concepts::Grid<TestGrid, ID>);

TEST_CASE("tiling::Grid::Grid", "[tiling::Grid]") {
    grid_test::test_constructors<TestGrid>(add_grid_width, add_grid_height);
}

TEST_CASE("tiling::Grid::copy_from_buffer", "[tiling::Grid]") {
    grid_test::test_copy_from_buffer<TestGrid>(add_grid_width, add_grid_height);
}

TEST_CASE("tiling::Grid::copy_to_buffer", "[tiling::Grid]") {
    grid_test::test_copy_to_buffer<TestGrid>(add_grid_width, add_grid_height);
}

TEST_CASE("tiling::Grid::make_similar", "[tiling::Grid]") {
    grid_test::test_make_similar<TestGrid>(add_grid_width, add_grid_height);
}

TEST_CASE("tiling::Grid::submit_read", "[tiling::Grid]") {
    buffer<ID, 2> in_buffer(range<2>(3 * tile_width, 3 * tile_height));
    {
        host_accessor in_buffer_ac(in_buffer, read_write);
        for (uindex_t c = 0; c < 3 * tile_width; c++) {
            for (uindex_t r = 0; r < 3 * tile_height; r++) {
                in_buffer_ac[c][r] = ID(c, r);
            }
        }
    }
    TestGrid grid(in_buffer);

    sycl::queue queue;

    using in_pipe = sycl::pipe<class tiled_grid_submit_read_test_id, ID>;
    grid.template submit_read<in_pipe>(queue, 1, 1);

    sycl::buffer<bool, 1> result_buffer = sycl::range<1>(1);
    queue.submit([&](sycl::handler &cgh) {
        accessor result_ac(result_buffer, cgh, write_only);
        uindex_t c_start = tile_width - halo_radius;
        uindex_t c_end = 2 * tile_width + halo_radius;
        uindex_t r_start = tile_height - halo_radius;
        uindex_t r_end = 2 * tile_height + halo_radius;

        cgh.single_task<class tiled_grid_submit_read_test_kernel>([=]() {
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

    REQUIRE(host_accessor(result_buffer)[0]);
}

TEST_CASE("tiling::Grid::submit_write", "[tiling::Grid]") {
    using out_pipe = sycl::pipe<class tiled_grid_submit_write_test_id, ID>;

    sycl::queue queue;
    queue.submit([&](sycl::handler &cgh) {
        cgh.single_task<class tiled_grid_submit_write_test_kernel>([=]() {
            for (uindex_t c = tile_width; c < 2 * tile_width; c++) {
                for (uindex_t r = tile_height; r < 2 * tile_height; r++) {
                    out_pipe::write(ID(c, r));
                }
            }
        });
    });

    TestGrid grid(3 * tile_width, 3 * tile_height);
    grid.template submit_write<out_pipe>(queue, 1, 1);

    sycl::buffer<ID, 2> out_buffer = sycl::range<2>(3 * tile_width, 3 * tile_height);
    grid.copy_to_buffer(out_buffer);

    host_accessor out_ac(out_buffer, read_only);
    for (uindex_t c = tile_width; c < 2 * tile_width; c++) {
        for (uindex_t r = tile_height; r < 2 * tile_height; r++) {
            REQUIRE(out_ac[c][r] == ID(c, r));
        }
    }
}
