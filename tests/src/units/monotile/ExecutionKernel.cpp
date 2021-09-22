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
    using TestExecutionKernel =
        monotile::ExecutionKernel<TransFunc, Cell, stencil_radius, pipeline_length, tile_width,
                                  tile_height, access::target::host_buffer>;

    buffer<Cell, 2> in_buffer(range<2>(tile_width, tile_height));
    buffer<Cell, 2> out_buffer(in_buffer.get_range());

    {
        auto ac = in_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < tile_width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                ac[c][r] = Cell{index_t(c), index_t(r), 0, CellStatus::Normal};
            }
        }
    }

    {
        auto in_ac = in_buffer.get_access<access::mode::read>();
        auto out_ac = out_buffer.get_access<access::mode::discard_write>();

        TestExecutionKernel(in_ac, out_ac, TransFunc(), 0, n_generations, Cell::halo())();
    }

    auto ac = out_buffer.get_access<access::mode::read>();
    for (uindex_t c = 0; c < tile_width; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            Cell cell = ac[c][r];
            REQUIRE(cell.c == c);
            REQUIRE(cell.r == r);
            REQUIRE(cell.i_generation == n_generations);
            REQUIRE(cell.status == CellStatus::Normal);
        }
    }
}

TEST_CASE("monotile::ExecutionKernel", "[monotile::ExecutionKernel]") {
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

    using TestExecutionKernel = monotile::ExecutionKernel<decltype(trans_func), Cell, 1, 16, 64, 64,
                                                          access::target::host_buffer>;

    buffer<Cell, 2> in_buffer(range<2>(64, 64));
    buffer<Cell, 2> out_buffer(range<2>(64, 64));
    {
        auto in_ac = in_buffer.get_access<access::mode::discard_write>();
        for (int c = 0; c < 64; c++) {
            for (int r = 0; r < 64; r++) {
                in_ac[c][r] = 0;
            }
        }
    }

    {
        auto in_ac = in_buffer.get_access<access::mode::read>();
        auto out_ac = out_buffer.get_access<access::mode::discard_write>();
        TestExecutionKernel kernel(in_ac, out_ac, trans_func, 16, 20, 0);
        kernel.operator()();
    }

    auto out_ac = out_buffer.get_access<access::mode::read>();
    for (int c = 0; c < 64; c++) {
        for (int r = 0; r < 64; r++) {
            REQUIRE(out_ac[c][r] == 4);
        }
    }
}
