/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
 * University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once
#include "TransFuncs.hpp"
#include <StencilStream/Concepts.hpp>

using namespace sycl;
using namespace stencil;

template <concepts::Grid<Cell> Grid, concepts::StencilUpdate<FPGATransFunc<1>, Grid> SU>
void test_stencil_update(stencil::uindex_t grid_width, uindex_t grid_height,
                         uindex_t iteration_offset, uindex_t n_iterations) {

    using Accessor = Grid::template GridAccessor<access::mode::read_write>;

    Grid input_grid(grid_width, grid_height);
    {
        Accessor ac(input_grid);
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                ac[c][r] =
                    Cell{index_t(c), index_t(r), (index_t)iteration_offset, 0, CellStatus::Normal};
            }
        }
    }

    SU update({.transition_function = FPGATransFunc<1>(),
               .halo_value = Cell::halo(),
               .iteration_offset = iteration_offset,
               .n_iterations = n_iterations});

    Grid output_grid = update(input_grid);

    Accessor ac(output_grid);
    for (uindex_t c = 0; c < grid_width; c++) {
        for (uindex_t r = 0; r < grid_height; r++) {
            REQUIRE(ac[c][r].c == c);
            REQUIRE(ac[c][r].r == r);
            REQUIRE(ac[c][r].i_iteration == iteration_offset + n_iterations);
            REQUIRE(ac[c][r].i_subiteration == 0);
            REQUIRE(ac[c][r].status == CellStatus::Normal);
        }
    }
}
