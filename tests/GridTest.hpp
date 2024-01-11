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
#pragma once
#include "constants.hpp"
#include <StencilStream/Concepts.hpp>
#include <StencilStream/Index.hpp>
#include <catch2/catch_all.hpp>

namespace grid_test {
template <stencil::concepts::Grid<stencil::ID> G>
void test_constructors(stencil::uindex_t grid_width, stencil::uindex_t grid_height) {
    G grid(1, 1);
    REQUIRE(grid.get_grid_width() == 1);
    REQUIRE(grid.get_grid_height() == 1);

    grid = G(tile_width, tile_height);
    REQUIRE(grid.get_grid_width() == tile_width);
    REQUIRE(grid.get_grid_height() == tile_height);

    grid = G(grid_width, grid_height);
    REQUIRE(grid.get_grid_width() == grid_width);
    REQUIRE(grid.get_grid_height() == grid_height);

    sycl::buffer<stencil::ID, 2> grid_buffer = sycl::range<2>(grid_width, grid_height);
    {
        sycl::host_accessor ac(grid_buffer, sycl::read_write);
        for (stencil::index_t c = 0; c < grid_width; c++) {
            for (stencil::index_t r = 0; r < grid_height; r++) {
                ac[c][r] = stencil::ID(c, r);
            }
        }
    }

    grid = grid_buffer;
    REQUIRE(grid.get_grid_width() == grid_width);
    REQUIRE(grid.get_grid_height() == grid_height);

    {
        typename G::template GridAccessor<sycl::access::mode::read> out_buffer_ac(grid);
        for (stencil::index_t c = 0; c < grid_width; c++) {
            for (stencil::index_t r = 0; r < grid_height; r++) {
                REQUIRE(out_buffer_ac[c][r] == stencil::ID(c, r));
            }
        }
    }
}

template <stencil::concepts::Grid<stencil::ID> G>
void test_copy_from_buffer(stencil::uindex_t grid_width, stencil::uindex_t grid_height) {
    sycl::buffer<stencil::ID, 2> in_buffer = sycl::range<2>(grid_width, grid_height);
    {
        sycl::host_accessor in_buffer_ac(in_buffer, sycl::read_write);
        for (stencil::index_t c = 0; c < grid_width; c++) {
            for (stencil::index_t r = 0; r < grid_height; r++) {
                in_buffer_ac[c][r] = stencil::ID(c, r);
            }
        }
    }

    G grid(grid_width, grid_height);
    grid.copy_from_buffer(in_buffer);

    {
        typename G::template GridAccessor<sycl::access::mode::read> grid_ac(grid);
        for (stencil::uindex_t c = 0; c < grid_width; c++) {
            for (stencil::uindex_t r = 0; r < grid_height; r++) {
                REQUIRE(grid_ac[c][r] == stencil::ID(c, r));
            }
        }
    }
}

template <stencil::concepts::Grid<stencil::ID> G>
void test_copy_to_buffer(stencil::uindex_t grid_width, stencil::uindex_t grid_height) {
    G grid(grid_width, grid_height);
    {
        typename G::template GridAccessor<sycl::access::mode::read_write> grid_ac(grid);
        for (stencil::uindex_t c = 0; c < grid_width; c++) {
            for (stencil::uindex_t r = 0; r < grid_height; r++) {
                grid_ac[c][r] = stencil::ID(c, r);
            }
        }
    }

    sycl::buffer<stencil::ID, 2> out_buffer = sycl::range<2>(grid_width, grid_height);
    grid.copy_to_buffer(out_buffer);
    {
        sycl::host_accessor ac(out_buffer, sycl::read_only);
        for (stencil::uindex_t c = 0; c < grid_width; c++) {
            for (stencil::uindex_t r = 0; r < grid_height; r++) {
                REQUIRE(ac[c][r] == stencil::ID(c, r));
            }
        }
    }
}

template <stencil::concepts::Grid<stencil::ID> G>
void test_make_similar(stencil::uindex_t grid_width, stencil::uindex_t grid_height) {
    G grid(grid_width, grid_height);
    G similar_grid = grid.make_similar();
    REQUIRE(similar_grid.get_grid_width() == grid_width);
    REQUIRE(similar_grid.get_grid_height() == grid_height);
}

} // namespace grid_test