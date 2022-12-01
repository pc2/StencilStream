/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include <CL/sycl.hpp>
#include <ext/intel/fpga_extensions.hpp>
#include <StencilStream/tiling/Grid.hpp>
#include <res/catch.hpp>
#include <res/constants.hpp>
#include <unordered_set>

using namespace stencil;
using namespace stencil::tiling;
using namespace cl::sycl;
using namespace std;

const uindex_t add_grid_width = grid_width + 1;
const uindex_t add_grid_height = grid_height + 1;

using TestGrid = Grid<ID, tile_width, tile_height, halo_radius, burst_length>;

TEST_CASE("Grid::Grid(uindex_t, uindex_t, T)", "[Grid]") {
    TestGrid grid(add_grid_width, add_grid_height);

    UID tile_range = grid.get_tile_range();
    REQUIRE(tile_range.c == add_grid_width / tile_width + 1);
    REQUIRE(tile_range.r == add_grid_height / tile_height + 1);
}

TEST_CASE("Grid::Grid(cl::sycl::buffer<T, 2>, T)", "[Grid]") {
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

    UID tile_range = grid.get_tile_range();
    REQUIRE(tile_range.c == add_grid_width / tile_width + 1);
    REQUIRE(tile_range.r == add_grid_height / tile_height + 1);

    buffer<ID, 2> out_buffer(range<2>(add_grid_width, add_grid_height));
    grid.copy_to(out_buffer);

    {
        auto out_buffer_ac = out_buffer.get_access<access::mode::read>();
        for (uindex_t c = 0; c < add_grid_width; c++) {
            for (uindex_t r = 0; r < add_grid_height; r++) {
                REQUIRE(out_buffer_ac[c][r].c == c);
                REQUIRE(out_buffer_ac[c][r].r == r);
            }
        }
    }
}

TEST_CASE("Grid::submit_tile_input", "[Grid]") {
    using grid_in_pipe = pipe<class grid_in_pipe_id, ID>;

    buffer<ID, 2> in_buffer(range<2>(3 * tile_width, 3 * tile_height));
    buffer<ID, 2> out_buffer(range<2>(2 * halo_radius + tile_width, 2 * halo_radius + tile_height));

#ifdef HARDWARE
    ext::intel::fpga_selector device_selector;
#else
    ext::intel::fpga_emulator_selector device_selector;
#endif
    cl::sycl::queue working_queue(device_selector);

    {
        auto in_buffer_ac = in_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < 3 * tile_width; c++) {
            for (uindex_t r = 0; r < 3 * tile_height; r++) {
                in_buffer_ac[c][r] = ID(c, r);
            }
        }
    }

    TestGrid grid(in_buffer);
    grid.submit_tile_input<grid_in_pipe>(working_queue, UID(1, 1));

    working_queue.submit([&](handler &cgh) {
        auto out_buffer_ac = out_buffer.get_access<access::mode::discard_write>(cgh);

        cgh.single_task<class input_test_kernel>([=]() {
            for (uindex_t c = 0; c < 2 * halo_radius + tile_width; c++) {
                for (uindex_t r = 0; r < 2 * halo_radius + tile_height; r++) {
                    out_buffer_ac[c][r] = grid_in_pipe::read();
                }
            }
        });
    });

    auto out_buffer_ac = out_buffer.get_access<access::mode::read>();

    for (uindex_t c = 0; c < 2 * halo_radius + tile_width; c++) {
        for (uindex_t r = 0; r < 2 * halo_radius + tile_height; r++) {
            REQUIRE(out_buffer_ac[c][r].c == c + tile_width - halo_radius);
            REQUIRE(out_buffer_ac[c][r].r == r + tile_height - halo_radius);
        }
    }
}

TEST_CASE("Grid::submit_tile_output", "[Grid]") {
    using grid_out_pipe = pipe<class grid_out_pipe_id, ID>;

    TestGrid grid(tile_width, tile_height);

#ifdef HARDWARE
    ext::intel::fpga_selector device_selector;
#else
    ext::intel::fpga_emulator_selector device_selector;
#endif
    cl::sycl::queue working_queue(device_selector);

    working_queue.submit([&](handler &cgh) {
        cgh.single_task<class output_test_kernel>([=]() {
            for (uindex_t c = 0; c < tile_width; c++) {
                for (uindex_t r = 0; r < tile_height; r++) {
                    grid_out_pipe::write(ID(c, r));
                }
            }
        });
    });

    grid.submit_tile_output<grid_out_pipe>(working_queue, UID(0, 0));

    buffer<ID, 2> out_buffer(range<2>(tile_width, tile_height));
    grid.copy_to(out_buffer);

    auto out_buffer_ac = out_buffer.get_access<access::mode::read>();
    for (uindex_t c = 0; c < tile_width; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            REQUIRE(out_buffer_ac[c][r].r == r);
            REQUIRE(out_buffer_ac[c][r].c == c);
        }
    }
}
