/*
 * Copyright © 2020-2022 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include <StencilStream/tiling/ExecutionKernel.hpp>
#include <res/HostPipe.hpp>
#include <res/TransFuncs.hpp>
#include <res/catch.hpp>
#include <res/constants.hpp>

using namespace stencil;
using namespace std;
using namespace stencil::tiling;
using namespace cl::sycl;

void test_tiling_kernel(uindex_t grid_width, uindex_t grid_height, uindex_t n_generations) {
    using TransFunc = FPGATransFunc<stencil_radius>;
    using in_pipe = HostPipe<class TilingExecutionKernelInPipeID, Cell>;
    using out_pipe = HostPipe<class TilingExecutionKernelOutPipeID, Cell>;
    using TestExecutionKernel = ExecutionKernel<TransFunc, n_processing_elements, tile_width,
                                                tile_height, in_pipe, out_pipe>;

    for (index_t c = -halo_radius; c < index_t(halo_radius + grid_width); c++) {
        for (index_t r = -halo_radius; r < index_t(halo_radius + grid_height); r++) {
            if (c >= index_t(0) && c < index_t(grid_width) && r >= index_t(0) &&
                r < index_t(grid_height)) {
                in_pipe::write(Cell{c, r, 0, CellStatus::Normal});
            } else {
                in_pipe::write(Cell::halo());
            }
        }
    }

    RunInformation run_information(0, n_generations);
    TestExecutionKernel(run_information, 0, n_generations, 0, 0, grid_width, grid_height,
                        Cell::halo())();

    buffer<Cell, 2> output_buffer(range<2>(grid_width, grid_height));

    {
        auto output_buffer_ac = output_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                output_buffer_ac[c][r] = out_pipe::read();
            }
        }
    }

    REQUIRE(in_pipe::empty());
    REQUIRE(out_pipe::empty());

    auto output_buffer_ac = output_buffer.get_access<access::mode::read>();
    for (uindex_t c = 1; c < grid_width; c++) {
        for (uindex_t r = 1; r < grid_height; r++) {
            Cell cell = output_buffer_ac[c][r];
            REQUIRE(cell.c == c);
            REQUIRE(cell.r == r);
            REQUIRE(cell.i_generation == n_generations);
            REQUIRE(cell.status == CellStatus::Normal);
        }
    }
}

TEST_CASE("tiling::ExecutionKernel", "[tiling::ExecutionKernel]") {
    test_tiling_kernel(tile_width, tile_height, n_processing_elements);
}

TEST_CASE("tiling::ExecutionKernel (partial tile)", "[tiling::ExecutionKernel]") {
    test_tiling_kernel(tile_width / 2, tile_height, n_processing_elements);
}

TEST_CASE("tiling::ExecutionKernel (partial pipeline)", "[tiling::ExecutionKernel]") {
    static_assert(n_processing_elements != 1);
    test_tiling_kernel(tile_width, tile_height, n_processing_elements - 1);
}

TEST_CASE("tiling::ExecutionKernel (noop)", "[tiling::ExecutionKernel]") {
    test_tiling_kernel(tile_width, tile_height, 0);
}

struct HaloHandlingKernel {
    using Cell = bool;
    static constexpr uindex_t stencil_radius = 1;

    struct IntermediateRepresentation {};

    HaloHandlingKernel(IntermediateRepresentation const &inter_rep, uindex_t i_generation) {}

    bool operator()(Stencil<HaloHandlingKernel> const &stencil) const {
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

TEST_CASE("Halo values inside the pipeline are handled correctly", "[tiling::ExecutionKernel]") {
    using in_pipe = HostPipe<class HaloValueTestInPipeID, bool>;
    using out_pipe = HostPipe<class HaloValueTestOutPipeID, bool>;
    using TestExecutionKernel = ExecutionKernel<HaloHandlingKernel, n_processing_elements,
                                                tile_width, tile_height, in_pipe, out_pipe>;

    uindex_t halo_radius = n_processing_elements;

    for (index_t c = -halo_radius; c < index_t(halo_radius + tile_width); c++) {
        for (index_t r = -halo_radius; r < index_t(halo_radius + tile_height); r++) {
            in_pipe::write(false);
        }
    }

    TestExecutionKernel(HaloHandlingKernel::IntermediateRepresentation{}, 0, n_processing_elements,
                        0, 0, tile_width, tile_height, true)();

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
    static constexpr uindex_t stencil_radius = 1;

    struct IntermediateRepresentation {};

    IncompletePipelineKernel(IntermediateRepresentation const &inter_rep, uindex_t i_generation) {}

    uint8_t operator()(Stencil<IncompletePipelineKernel> const &stencil) const {
        return stencil[ID(0, 0)] + 1;
    }
};

TEST_CASE("Incomplete Pipeline with i_generation != 0", "[tiling::ExecutionKernel]") {
    using in_pipe = HostPipe<class IncompletePipelineInPipeID, uint8_t>;
    using out_pipe = HostPipe<class IncompletePipelineOutPipeID, uint8_t>;
    using TestExecutionKernel =
        ExecutionKernel<IncompletePipelineKernel, 16, 64, 64, in_pipe, out_pipe>;

    for (int c = -16; c < 16 + 64; c++) {
        for (int r = -16; r < 16 + 64; r++) {
            in_pipe::write(0);
        }
    }

    TestExecutionKernel kernel(IncompletePipelineKernel::IntermediateRepresentation{}, 16, 20, 0, 0,
                               64, 64, 0);
    kernel.operator()();

    REQUIRE(in_pipe::empty());

    for (int c = 0; c < 64; c++) {
        for (int r = 0; r < 64; r++) {
            REQUIRE(out_pipe::read() == 4);
        }
    }

    REQUIRE(out_pipe::empty());
}