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
#include "../TransFuncs.hpp"
#include "../constants.hpp"
#include <StencilStream/cpu/StencilUpdate.hpp>
#include <StencilStream/tdv/InlineSupplier.hpp>

using namespace sycl;
using namespace stencil;
using namespace stencil::cpu;

using StencilUpdateImpl = StencilUpdate<FPGATransFunc<1>, tdv::InlineSupplier<GenerationFunction>>;
using GridImpl = typename StencilUpdateImpl::GridImpl;

TEST_CASE("cpu::StencilUpdate", "[cpu::StencilUpdate]") {
    uindex_t grid_width = 64;
    uindex_t grid_height = 64;
    uindex_t n_generations = 64;

    GridImpl input_grid(grid_width, grid_height);
    {
        GridImpl::GridAccessor<access::mode::read_write> ac(input_grid);
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                ac[c][r] = Cell{index_t(c), index_t(r), 0, 0, CellStatus::Normal};
            }
        }
    }

    StencilUpdateImpl update(
        {.transition_function = FPGATransFunc<1>(),
         .halo_value = Cell::halo(),
         .n_generations = n_generations,
         .tdv_host_state = tdv::InlineSupplier<GenerationFunction>(GenerationFunction())});

    GridImpl output_grid = update(input_grid);

    GridImpl::GridAccessor<access::mode::read> ac(output_grid);
    for (uindex_t c = 0; c < grid_width; c++) {
        for (uindex_t r = 0; r < grid_height; r++) {
            REQUIRE(ac[c][r].c == c);
            REQUIRE(ac[c][r].r == r);
            REQUIRE(ac[c][r].i_generation == n_generations);
            REQUIRE(ac[c][r].i_subgeneration == 0);
            REQUIRE(ac[c][r].status == CellStatus::Normal);
        }
    }
}