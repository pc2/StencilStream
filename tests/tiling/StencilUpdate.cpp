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
#include <StencilStream/tiling/StencilUpdate.hpp>

using namespace cl::sycl;
using namespace stencil;
using namespace stencil::tiling;

using StencilUpdateImpl =
    StencilUpdate<FPGATransFunc<1>, n_processing_elements, tile_width, tile_height>;
using GridImpl = typename StencilUpdateImpl::GridImpl;

void test_tiling_stencil_update(uindex_t grid_width, uindex_t grid_height, uindex_t n_generations) {
    buffer<Cell, 2> input_buffer(range<2>(grid_width, grid_height));
    {
        auto in_buffer_ac = input_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                in_buffer_ac[c][r] = Cell{index_t(c), index_t(r), 0, 0, CellStatus::Normal};
            }
        }
    }
    GridImpl input_grid = input_buffer;

    StencilUpdateImpl update({
        .transition_function = FPGATransFunc<1>(),
        .halo_value = Cell::halo(),
        .n_generations = n_generations,
    });

    GridImpl output_grid = update(input_grid);

    buffer<Cell, 2> output_buffer(range<2>(grid_width, grid_height));
    output_grid.copy_to_buffer(output_buffer);

    auto out_buffer_ac = output_buffer.get_access<access::mode::read>();
    for (uindex_t c = 0; c < grid_width; c++) {
        for (uindex_t r = 0; r < grid_height; r++) {
            REQUIRE(out_buffer_ac[c][r].c == c);
            REQUIRE(out_buffer_ac[c][r].r == r);
            REQUIRE(out_buffer_ac[c][r].i_generation == n_generations);
            REQUIRE(out_buffer_ac[c][r].i_subgeneration == 0);
            REQUIRE(out_buffer_ac[c][r].status == CellStatus::Normal);
        }
    }
}

TEST_CASE("tiling::StencilUpdate", "[tiling::StencilUpdate]") {
    for (uindex_t i_grid_width = 0; i_grid_width < 3; i_grid_width++) {
        for (uindex_t i_grid_height = 0; i_grid_height < 3; i_grid_height++) {
            uindex_t grid_width = (1 + i_grid_width) * (tile_width / 2);
            uindex_t grid_height = (1 + i_grid_height) * (tile_height / 2);

            for (index_t gen_offset = -1; gen_offset <= 1; gen_offset++) {
                test_tiling_stencil_update(grid_width, grid_height, gens_per_pass + gen_offset);
            }
        }
    }
}