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
#include <StencilStream/monotile/ExecutionKernel.hpp>
#include <res/HostPipe.hpp>
#include <res/TransFuncs.hpp>
#include <res/catch.hpp>
#include <res/constants.hpp>

using namespace stencil;
using namespace std;
using namespace cl::sycl;

void test_monotile_kernel(uindex_t grid_width, uindex_t grid_height, uindex_t n_generations) {
    using TransFunc = HostTransFunc<stencil_radius>;
    using in_pipe = HostPipe<class MonotileExecutionKernelInPipeID, Cell>;
    using out_pipe = HostPipe<class MonotileExecutionKernelOutPipeID, Cell>;
    using TestExecutionKernel =
        monotile::ExecutionKernel<TransFunc, n_processing_elements, tile_width, tile_height,
                                  in_pipe, out_pipe>;

    for (uindex_t c = 0; c < grid_width; c++) {
        for (uindex_t r = 0; r < grid_height; r++) {
            in_pipe::write(Cell{index_t(c), index_t(r), 0, CellStatus::Normal});
        }
    }

    RunInformation iter_rep{0, n_generations - 1};

    TestExecutionKernel(iter_rep, 0, n_generations, grid_width, grid_height, Cell::halo())();

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

TEST_CASE("monotile::ExecutionKernel", "[monotile::ExecutionKernel]") {
    test_monotile_kernel(tile_width, tile_height, n_processing_elements);
}

TEST_CASE("monotile::ExecutionKernel (partial tile)", "[monotile::ExecutionKernel]") {
    test_monotile_kernel(tile_width / 2, tile_height / 2, n_processing_elements);
}

TEST_CASE("monotile::ExecutionKernel (partial pipeline)", "[monotile::ExecutionKernel]") {
    static_assert(n_processing_elements != 1);
    test_monotile_kernel(tile_width, tile_height, n_processing_elements - 1);
}

TEST_CASE("monotile::ExecutionKernel (noop)", "[monotile::ExecutionKernel]") {
    test_monotile_kernel(tile_width, tile_height, 0);
}

struct IncompletePipelineKernel {
    using Cell = uint8_t;
    static constexpr uindex_t stencil_radius = 1;

    struct IntermediateRepresentation {};

    IncompletePipelineKernel(IntermediateRepresentation const &inter_rep, uindex_t i_generation) {}

    Cell operator()(Stencil<IncompletePipelineKernel> const &stencil) const {
        return stencil[ID(0, 0)] + 1;
    }
};

TEST_CASE("monotile::ExecutionKernel: Incomplete Pipeline with i_generation != 0",
          "[monotile::ExecutionKernel]") {

    using in_pipe = HostPipe<class IncompletePipelineInPipeID, uint8_t>;
    using out_pipe = HostPipe<class IncompletePipelineOutPipeID, uint8_t>;
    using TestExecutionKernel =
        monotile::ExecutionKernel<IncompletePipelineKernel, 16, 64, 64, in_pipe, out_pipe>;

    for (int c = 0; c < 64; c++) {
        for (int r = 0; r < 64; r++) {
            in_pipe::write(0);
        }
    }

    TestExecutionKernel kernel(IncompletePipelineKernel::IntermediateRepresentation{}, 16, 20, 64,
                               64, 0);
    kernel.operator()();

    REQUIRE(in_pipe::empty());

    for (int c = 0; c < 64; c++) {
        for (int r = 0; r < 64; r++) {
            REQUIRE(out_pipe::read() == 4);
        }
    }

    REQUIRE(out_pipe::empty());
}