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
#include <StencilStream/monotile/ExecutionKernel.hpp>
#include <res/HostPipe.hpp>
#include <res/TransFuncs.hpp>
#include <res/catch.hpp>
#include <res/constants.hpp>

using namespace stencil;
using namespace std;
using namespace cl::sycl;

void test_monotile_kernel(uindex_t n_generations) {
    using TransFunc = HostTransFunc<stencil_radius>;
    using in_pipe = HostPipe<class MonotileExecutionKernelInPipeID, Cell>;
    using out_pipe = HostPipe<class MonotileExecutionKernelOutPipeID, Cell>;
    using TestExecutionKernel =
        monotile::ExecutionKernel<TransFunc, Cell, stencil_radius, pipeline_length, tile_width,
                                tile_height, in_pipe, out_pipe>;

    for (uindex_t c = 0; c < tile_width; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            in_pipe::write(Cell{index_t(c), index_t(r), 0, CellStatus::Normal});
        }
    }

    TestExecutionKernel(TransFunc(), 0, n_generations, tile_width, tile_height, Cell::halo())();

    buffer<Cell, 2> output_buffer(range<2>(tile_width, tile_height));

    {
        auto output_buffer_ac = output_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < tile_width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                output_buffer_ac[c][r] = out_pipe::read();
            }
        }
    }

    REQUIRE(in_pipe::empty());
    REQUIRE(out_pipe::empty());

    auto output_buffer_ac = output_buffer.get_access<access::mode::read>();
    for (uindex_t c = 1; c < tile_width; c++) {
        for (uindex_t r = 1; r < tile_height; r++) {
            Cell cell = output_buffer_ac[c][r];
            REQUIRE(cell.c == c);
            REQUIRE(cell.r == r);
            REQUIRE(cell.i_generation == n_generations);
            REQUIRE(cell.status == CellStatus::Normal);
        }
    }
}

TEST_CASE("monotile::ExecutionKernel", "[monotile::ExecutionKernel]") {
    ac_int<8> my_int = 42;
    REQUIRE(my_int[1] == 1);
    test_monotile_kernel(pipeline_length);
}

TEST_CASE("monotile::ExecutionKernel (partial pipeline)", "[monotile::ExecutionKernel]") {
    static_assert(pipeline_length != 1);
    test_monotile_kernel(pipeline_length - 1);
}

TEST_CASE("monotile::ExecutionKernel (noop)", "[monotile::ExecutionKernel]") {
    test_monotile_kernel(0);
}

TEST_CASE("monotile::ExecutionKernel: Incomplete Pipeline with i_generation != 0",
          "[monotile::ExecutionKernel]") {
    using Cell = uint8_t;
    auto trans_func = [](Stencil<Cell, 1> const &stencil) { return stencil[ID(0, 0)] + 1; };

    using in_pipe = HostPipe<class IncompletePipelineInPipeID, Cell>;
    using out_pipe = HostPipe<class IncompletePipelineOutPipeID, Cell>;
    using TestExecutionKernel =
        monotile::ExecutionKernel<decltype(trans_func), Cell, 1, 16, 64, 64, in_pipe, out_pipe>;

    for (int c = 0; c < 64; c++) {
        for (int r = 0; r < 64; r++) {
            in_pipe::write(0);
        }
    }

    TestExecutionKernel kernel(trans_func, 16, 20, 64, 64, 0);
    kernel.operator()();

    REQUIRE(in_pipe::empty());

    for (int c = 0; c < 64; c++) {
        for (int r = 0; r < 64; r++) {
            REQUIRE(out_pipe::read() == 4);
        }
    }

    REQUIRE(out_pipe::empty());
}