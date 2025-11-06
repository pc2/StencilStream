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
#include <StencilStream/Concepts.hpp>
#include <catch2/catch_all.hpp>

namespace grid_test {

template <typename Cell, stencil::concepts::Grid<Cell> G>
void test_constructors(std::size_t grid_height, std::size_t grid_width) {
    G grid(1, 1);
    REQUIRE(grid.get_grid_height() == 1);
    REQUIRE(grid.get_grid_width() == 1);

    grid = G(grid_height / 2, grid_width / 2);
    REQUIRE(grid.get_grid_height() == grid_height / 2);
    REQUIRE(grid.get_grid_width() == grid_width / 2);

    grid = G(grid_height, grid_width);
    REQUIRE(grid.get_grid_height() == grid_height);
    REQUIRE(grid.get_grid_width() == grid_width);

    sycl::buffer<Cell, 2> grid_buffer = sycl::range<2>(grid_height, grid_width);
    {
        sycl::host_accessor ac(grid_buffer, sycl::read_write);
        for (std::size_t r = 0; r < grid_height; r++) {
            for (std::size_t c = 0; c < grid_width; c++) {
                ac[r][c].id = sycl::id<2>(r, c);
            }
        }
    }

    grid = grid_buffer;
    REQUIRE(grid.get_grid_height() == grid_height);
    REQUIRE(grid.get_grid_width() == grid_width);

    {
        typename G::template GridAccessor<sycl::access::mode::read> out_buffer_ac(grid);
        for (std::size_t r = 0; r < grid_height; r++) {
            for (std::size_t c = 0; c < grid_width; c++) {
                REQUIRE(out_buffer_ac[r][c].id == sycl::id<2>(r, c));
            }
        }
    }
}

template <typename Cell, stencil::concepts::Grid<Cell> G>
void test_copy_from_buffer(std::size_t grid_height, std::size_t grid_width) {
    sycl::buffer<Cell, 2> in_buffer = sycl::range<2>(grid_height, grid_width);
    {
        sycl::host_accessor in_buffer_ac(in_buffer, sycl::read_write);
        for (std::size_t r = 0; r < grid_height; r++) {
            for (std::size_t c = 0; c < grid_width; c++) {
                in_buffer_ac[r][c].id = sycl::id<2>(r, c);
            }
        }
    }

    G grid(grid_height, grid_width);
    grid.copy_from_buffer(in_buffer);

    {
        typename G::template GridAccessor<sycl::access::mode::read> grid_ac(grid);
        for (std::size_t r = 0; r < grid_height; r++) {
            for (std::size_t c = 0; c < grid_width; c++) {
                REQUIRE(grid_ac[r][c].id == sycl::id<2>(r, c));
            }
        }
    }
}

template <typename Cell, stencil::concepts::Grid<Cell> G>
void test_copy_to_buffer(std::size_t grid_height, std::size_t grid_width) {
    G grid(grid_height, grid_width);
    {
        typename G::template GridAccessor<sycl::access::mode::read_write> grid_ac(grid);
        for (std::size_t r = 0; r < grid_height; r++) {
            for (std::size_t c = 0; c < grid_width; c++) {
                grid_ac[r][c].id = sycl::id<2>(r, c);
            }
        }
    }

    sycl::buffer<Cell, 2> out_buffer = sycl::range<2>(grid_height, grid_width);
    grid.copy_to_buffer(out_buffer);
    {
        sycl::host_accessor ac(out_buffer, sycl::read_only);
        for (std::size_t r = 0; r < grid_height; r++) {
            for (std::size_t c = 0; c < grid_width; c++) {
                REQUIRE(ac[r][c].id == sycl::id<2>(r, c));
            }
        }
    }
}

template <typename Cell, stencil::concepts::Grid<Cell> G>
void test_make_similar(std::size_t grid_height, std::size_t grid_width) {
    G grid(grid_height, grid_width);
    G similar_grid = grid.make_similar();
    REQUIRE(similar_grid.get_grid_height() == grid_height);
    REQUIRE(similar_grid.get_grid_width() == grid_width);
}

} // namespace grid_test