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
#include "../HostPipe.hpp"
#include "../TransFuncs.hpp"
#include "../constants.hpp"
#include <StencilStream/monotile/StencilUpdate.hpp>
#include <StencilStream/tdv/InlineSupplier.hpp>

using namespace sycl;
using namespace stencil;
using namespace stencil::monotile;

constexpr TileParameters tile_params{.width = tile_width, .height = tile_height};
constexpr StencilUpdateParameters su_params{.n_processing_elements = n_processing_elements};
using StencilUpdateImpl = StencilUpdate<FPGATransFunc<1>, tdv::InlineSupplier<GenerationFunction>,
                                        tile_params, su_params>;
using GridImpl = typename StencilUpdateImpl::GridImpl;

void test_monotile_kernel(uindex_t grid_width, uindex_t grid_height, uindex_t target_i_generation) {
    using TransFunc = HostTransFunc<stencil_radius>;
    using in_pipe = HostPipe<class MonotileStencilUpdateKernelInPipeID, Cell>;
    using out_pipe = HostPipe<class MonotileStencilUpdateKernelOutPipeID, Cell>;
    using KernelArgument = tdv::InlineSupplier<GenerationFunction>::KernelArgument;
    using TestKernel =
        StencilUpdateKernel<TransFunc, KernelArgument, tile_params, su_params, in_pipe, out_pipe>;

    for (uindex_t c = 0; c < grid_width; c++) {
        for (uindex_t r = 0; r < grid_height; r++) {
            in_pipe::write(Cell{index_t(c), index_t(r), 0, 0, CellStatus::Normal});
        }
    }

    TestKernel(TransFunc(), 0, target_i_generation, grid_width, grid_height, Cell::halo(),
               KernelArgument{.function = GenerationFunction{}, .i_generation = 0})();

    buffer<Cell, 2> output_buffer(range<2>(grid_width, grid_height));

    {
        host_accessor output_buffer_ac(output_buffer, write_only);
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                output_buffer_ac[c][r] = out_pipe::read();
            }
        }
    }

    REQUIRE(in_pipe::empty());
    REQUIRE(out_pipe::empty());

    host_accessor output_buffer_ac(output_buffer, read_only);
    for (uindex_t c = 1; c < grid_width; c++) {
        for (uindex_t r = 1; r < grid_height; r++) {
            Cell cell = output_buffer_ac[c][r];
            REQUIRE(cell.c == c);
            REQUIRE(cell.r == r);
            REQUIRE(cell.i_generation == target_i_generation);
            REQUIRE(cell.i_subgeneration == 0);
            REQUIRE(cell.status == CellStatus::Normal);
        }
    }
}

TEST_CASE("monotile::StencilUpdateKernel", "[monotile::StencilUpdateKernel]") {
    test_monotile_kernel(tile_width, tile_height, gens_per_pass);
}

TEST_CASE("monotile::StencilUpdateKernel (partial tile)", "[monotile::StencilUpdateKernel]") {
    test_monotile_kernel(tile_width / 2, tile_height / 2, gens_per_pass);
}

TEST_CASE("monotile::StencilUpdateKernel (partial pipeline)", "[monotile::StencilUpdateKernel]") {
    static_assert(gens_per_pass != 1);
    test_monotile_kernel(tile_width, tile_height, gens_per_pass - 1);
}

TEST_CASE("monotile::StencilUpdateKernel (noop)", "[monotile::StencilUpdateKernel]") {
    test_monotile_kernel(tile_width, tile_height, 0);
}

struct IncompletePipelineKernel {
    using Cell = uint8_t;
    using TimeDependentValue = std::monostate;

    static constexpr uindex_t stencil_radius = 1;
    static constexpr uindex_t n_subgenerations = 1;

    Cell operator()(Stencil<uint8_t, 1> const &stencil) const { return stencil[ID(0, 0)] + 1; }
};

TEST_CASE("monotile::StencilUpdateKernel: Incomplete Pipeline with i_generation != 0",
          "[monotile::StencilUpdateKernel]") {

    using in_pipe = HostPipe<class IncompletePipelineInPipeID, uint8_t>;
    using out_pipe = HostPipe<class IncompletePipelineOutPipeID, uint8_t>;
    using TestKernel = StencilUpdateKernel<
        IncompletePipelineKernel, tdv::NoneSupplier, TileParameters{.width = 64, .height = 64},
        StencilUpdateParameters{.n_processing_elements = 16}, in_pipe, out_pipe>;

    for (int c = 0; c < 64; c++) {
        for (int r = 0; r < 64; r++) {
            in_pipe::write(0);
        }
    }

    TestKernel kernel(IncompletePipelineKernel(), 16, 20, 64, 64, 0, tdv::NoneSupplier{});
    kernel.operator()();

    REQUIRE(in_pipe::empty());

    for (int c = 0; c < 64; c++) {
        for (int r = 0; r < 64; r++) {
            REQUIRE(out_pipe::read() == 4);
        }
    }

    REQUIRE(out_pipe::empty());
}

void test_monotile_stencil_update(uindex_t grid_width, uindex_t grid_height,
                                  uindex_t n_generations) {
    buffer<Cell, 2> input_buffer(range<2>(grid_width, grid_height));
    {
        host_accessor in_buffer_ac(input_buffer, read_write);
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
        .tdv_host_state = tdv::InlineSupplier<GenerationFunction>(GenerationFunction()),
    });

    GridImpl output_grid = update(input_grid);

    buffer<Cell, 2> output_buffer(range<2>(grid_width, grid_height));
    output_grid.copy_to_buffer(output_buffer);

    host_accessor out_buffer_ac(output_buffer, read_only);
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

TEST_CASE("monotile::StencilUpdate", "[monotile::StencilUpdate]") {
    for (uindex_t grid_width = tile_width / 2; grid_width < tile_width; grid_width += 1) {
        for (uindex_t grid_height = tile_height / 2; grid_height < tile_height; grid_height += 1) {
            for (index_t gen_offset = -1; gen_offset <= 1; gen_offset++) {
                test_monotile_stencil_update(grid_width, grid_height, gens_per_pass + gen_offset);
            }
        }
    }
}