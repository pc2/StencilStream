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
#include <StencilStream/tdv/InlineSupplier.hpp>
#include <StencilStream/tiling/StencilUpdate.hpp>

using namespace sycl;
using namespace stencil;
using namespace stencil::tiling;

constexpr TileParameters default_tile_params = {
    .width = tile_width,
    .height = tile_height,
    .halo_radius = halo_radius,
};
constexpr StencilUpdateParameters default_su_params = {
    .n_processing_elements = n_processing_elements,
};

void test_tiling_kernel(uindex_t grid_width, uindex_t grid_height, uindex_t target_i_generation) {
    using TransFunc = FPGATransFunc<stencil_radius>;
    using in_pipe = HostPipe<class TilingStencilUpdateKernelInPipeID, Cell>;
    using out_pipe = HostPipe<class TilingStencilUpdateKernelOutPipeID, Cell>;
    using KernelArgument = tdv::InlineSupplier<GenerationFunction>::KernelArgument;
    using TestKernel = StencilUpdateKernel<TransFunc, KernelArgument, default_tile_params,
                                           default_su_params, in_pipe, out_pipe>;

    for (index_t c = -halo_radius; c < index_t(halo_radius + grid_width); c++) {
        for (index_t r = -halo_radius; r < index_t(halo_radius + grid_height); r++) {
            if (c >= index_t(0) && c < index_t(grid_width) && r >= index_t(0) &&
                r < index_t(grid_height)) {
                in_pipe::write(Cell{c, r, 0, 0, CellStatus::Normal});
            } else {
                in_pipe::write(Cell::halo());
            }
        }
    }

    TestKernel(TransFunc(), 0, target_i_generation, 0, 0, grid_width, grid_height, Cell::halo(),
               KernelArgument{.function = GenerationFunction{}, .i_generation = 0})();

    buffer<Cell, 2> output_buffer(range<2>(grid_width, grid_height));

    {
        host_accessor output_buffer_ac(output_buffer, read_write);
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

TEST_CASE("tiling::StencilUpdateKernel", "[tiling::StencilUpdateKernel]") {
    test_tiling_kernel(tile_width, tile_height, gens_per_pass);
}

TEST_CASE("tiling::StencilUpdateKernel (partial tile)", "[tiling::StencilUpdateKernel]") {
    test_tiling_kernel(tile_width / 2, tile_height, gens_per_pass);
}

TEST_CASE("tiling::StencilUpdateKernel (partial pipeline)", "[tiling::StencilUpdateKernel]") {
    static_assert(gens_per_pass != 1);
    test_tiling_kernel(tile_width, tile_height, gens_per_pass - 1);
}

TEST_CASE("tiling::StencilUpdateKernel (noop)", "[tiling::StencilUpdateKernel]") {
    test_tiling_kernel(tile_width, tile_height, 0);
}

struct HaloHandlingKernel {
    using Cell = bool;
    using TimeDependentValue = std::monostate;

    static constexpr uindex_t stencil_radius = 1;
    static constexpr stencil::uindex_t n_subgenerations = 1;

    bool operator()(Stencil<bool, 1> const &stencil) const {
        ID idx = stencil.id;
        bool is_valid = true;
        if (idx.c == 0) {
            is_valid &= stencil[ID(-1, -1)] && stencil[ID(-1, 0)] && stencil[ID(-1, 1)];
        } else if (idx.c == tile_width - 1) {
            is_valid &= stencil[ID(1, -1)] && stencil[ID(1, 0)] && stencil[ID(1, 1)];
        }

        if (idx.r == 0) {
            is_valid &= stencil[ID(-1, -1)] && stencil[ID(0, -1)] && stencil[ID(1, -1)];
        } else if (idx.r == tile_height - 1) {
            is_valid &= stencil[ID(-1, 1)] && stencil[ID(0, 1)] && stencil[ID(1, 1)];
        }

        return is_valid;
    }
};

TEST_CASE("Halo values inside the pipeline are handled correctly",
          "[tiling::StencilUpdateKernel]") {
    using in_pipe = HostPipe<class HaloValueTestInPipeID, bool>;
    using out_pipe = HostPipe<class HaloValueTestOutPipeID, bool>;
    using TestKernel = StencilUpdateKernel<HaloHandlingKernel, tdv::NoneSupplier,
                                           TileParameters{.width = tile_width,
                                                          .height = tile_height,
                                                          .halo_radius = n_processing_elements},
                                           default_su_params, in_pipe, out_pipe>;

    uindex_t halo_radius = n_processing_elements;

    for (index_t c = -halo_radius; c < index_t(halo_radius + tile_width); c++) {
        for (index_t r = -halo_radius; r < index_t(halo_radius + tile_height); r++) {
            in_pipe::write(false);
        }
    }

    TestKernel(HaloHandlingKernel(), 0, gens_per_pass, 0, 0, tile_width, tile_height, true,
               tdv::NoneSupplier())();

    for (uindex_t c = 0; c < tile_width; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            REQUIRE(out_pipe::read());
        }
    }

    REQUIRE(in_pipe::empty());
    REQUIRE(out_pipe::empty());
}

struct IncompletePipelineKernel {
    using Cell = uint8_t;
    using TimeDependentValue = std::monostate;

    static constexpr uindex_t stencil_radius = 1;
    static constexpr stencil::uindex_t n_subgenerations = 1;

    uint8_t operator()(Stencil<uint8_t, 1> const &stencil) const { return stencil[ID(0, 0)] + 1; }
};

TEST_CASE("Incomplete Pipeline with i_generation != 0", "[tiling::StencilUpdateKernel]") {
    using in_pipe = HostPipe<class IncompletePipelineInPipeID, uint8_t>;
    using out_pipe = HostPipe<class IncompletePipelineOutPipeID, uint8_t>;
    using TestKernel =
        StencilUpdateKernel<IncompletePipelineKernel, tdv::NoneSupplier,
                            TileParameters{.width = 64, .height = 64, .halo_radius = 16},
                            StencilUpdateParameters{.n_processing_elements = 16}, in_pipe,
                            out_pipe>;

    for (int c = -16; c < 16 + 64; c++) {
        for (int r = -16; r < 16 + 64; r++) {
            in_pipe::write(0);
        }
    }

    TestKernel kernel(IncompletePipelineKernel(), 16, 20, 0, 0, 64, 64, 0, tdv::NoneSupplier());
    kernel.operator()();

    REQUIRE(in_pipe::empty());

    for (int c = 0; c < 64; c++) {
        for (int r = 0; r < 64; r++) {
            REQUIRE(out_pipe::read() == 4);
        }
    }

    REQUIRE(out_pipe::empty());
}

void test_tiling_stencil_update(uindex_t grid_width, uindex_t grid_height, uindex_t n_generations) {
    using StencilUpdateImpl =
        StencilUpdate<FPGATransFunc<stencil_radius>, tdv::InlineSupplier<GenerationFunction>,
                      default_tile_params, default_su_params>;
    using GridImpl = Grid<Cell, default_tile_params>;

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
        .transition_function = FPGATransFunc<stencil_radius>(),
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