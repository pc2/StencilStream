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
#include <StencilStream/tiling/ExecutionKernel.hpp>
#include <res/HostPipe.hpp>
#include <res/TransFuncs.hpp>
#include <res/catch.hpp>
#include <res/constants.hpp>

using namespace stencil;
using namespace std;
using namespace stencil::tiling;
using namespace cl::sycl;

void test_tiling_kernel(uindex_t n_generations) {
    using TransFunc = FPGATransFunc<stencil_radius>;
    using in_pipe_0 = HostPipe<class TilingExecutionKernelInPipe0ID, Cell>;
    using in_pipe_1 = HostPipe<class TilingExecutionKernelInPipe1ID, Cell>;
    using in_pipe_2 = HostPipe<class TilingExecutionKernelInPipe2ID, Cell>;
    using in_pipe_3 = HostPipe<class TilingExecutionKernelInPipe3ID, Cell>;
    using in_pipe_4 = HostPipe<class TilingExecutionKernelInPipe4ID, Cell>;
    using out_pipe_0 = HostPipe<class TilingExecutionKernelOutPipe0ID, Cell>;
    using out_pipe_1 = HostPipe<class TilingExecutionKernelOutPipe1ID, Cell>;
    using out_pipe_2 = HostPipe<class TilingExecutionKernelOutPipe2ID, Cell>;
    using TestExecutionKernel = ExecutionKernel<TransFunc, Cell, stencil_radius, pipeline_length,
                                                tile_width, tile_height, in_pipe_0, in_pipe_1,
                                                in_pipe_2, in_pipe_3, in_pipe_4, out_pipe_0,
                                                out_pipe_1, out_pipe_2>;

    for (index_t c = -halo_radius; c < index_t(halo_radius + tile_width); c++) {
        for (index_t r = -halo_radius; r < index_t(halo_radius + tile_height); r++) {
            Cell value;
            if (c >= index_t(0) && c < index_t(tile_width) && r >= index_t(0) &&
                r < index_t(tile_height)) {
                value = Cell{c, r, 0, CellStatus::Normal};
            } else {
                value = Cell::halo();
            }
            if (r < 0) {
                in_pipe_0::write(value);
            } else if (r < halo_radius) {
                in_pipe_1::write(value);
            } else if (r < halo_radius + core_height) {
                in_pipe_2::write(value);
            } else if (r < tile_height) {
                in_pipe_3::write(value);
            } else {
                in_pipe_4::write(value);
            }
        }
    }

    TestExecutionKernel(TransFunc(), 0, n_generations, 0, 0, tile_width, tile_height,
                        Cell::halo())();

    buffer<Cell, 2> output_buffer(range<2>(tile_width, tile_height));

    {
        auto output_buffer_ac = output_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < tile_width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                Cell value;
                if (r < halo_radius) {
                    value = out_pipe_0::read();
                } else if (r < halo_radius + core_height) {
                    value = out_pipe_1::read();
                } else {
                    value = out_pipe_2::read();
                }
                output_buffer_ac[c][r] = value;
            }
        }
    }

    REQUIRE(in_pipe_0::empty());
    REQUIRE(in_pipe_1::empty());
    REQUIRE(in_pipe_2::empty());
    REQUIRE(in_pipe_3::empty());
    REQUIRE(in_pipe_4::empty());
    REQUIRE(out_pipe_0::empty());
    REQUIRE(out_pipe_1::empty());
    REQUIRE(out_pipe_2::empty());

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

TEST_CASE("tiling::ExecutionKernel", "[tiling::ExecutionKernel]") {
    test_tiling_kernel(pipeline_length);
}

TEST_CASE("tiling::ExecutionKernel (partial pipeline)", "[tiling::ExecutionKernel]") {
    static_assert(pipeline_length != 1);
    test_tiling_kernel(pipeline_length - 1);
}

TEST_CASE("tiling::ExecutionKernel (noop)", "[tiling::ExecutionKernel]") { test_tiling_kernel(0); }

TEST_CASE("Halo values inside the pipeline are handled correctly", "[tiling::ExecutionKernel]") {
    auto my_kernel = [=](Stencil<bool, stencil_radius> const &stencil) {
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
    };

    using in_pipe_0 = HostPipe<class HaloValueTestInPipe0ID, bool>;
    using in_pipe_1 = HostPipe<class HaloValueTestInPipe1ID, bool>;
    using in_pipe_2 = HostPipe<class HaloValueTestInPipe2ID, bool>;
    using in_pipe_3 = HostPipe<class HaloValueTestInPipe3ID, bool>;
    using in_pipe_4 = HostPipe<class HaloValueTestInPipe4ID, bool>;
    using out_pipe_0 = HostPipe<class HaloValueTestOutPipe0ID, bool>;
    using out_pipe_1 = HostPipe<class HaloValueTestOutPipe1ID, bool>;
    using out_pipe_2 = HostPipe<class HaloValueTestOutPipe2ID, bool>;
    using TestExecutionKernel =
        ExecutionKernel<decltype(my_kernel), bool, stencil_radius, pipeline_length, tile_width,
                        tile_height, in_pipe_0, in_pipe_1, in_pipe_2, in_pipe_3, in_pipe_4, 
                        out_pipe_0, out_pipe_1, out_pipe_2>;

    for (index_t c = -halo_radius; c < index_t(halo_radius + tile_width); c++) {
        for (index_t r = -halo_radius; r < index_t(halo_radius + tile_height); r++) {
            if (r < 0) {
                in_pipe_0::write(false);
            } else if (r < halo_radius) {
                in_pipe_1::write(false);
            } else if (r < halo_radius + core_height) {
                in_pipe_2::write(false);
            } else if (r < tile_height) {
                in_pipe_3::write(false);
            } else {
                in_pipe_4::write(false);
            }
        }
    }

    TestExecutionKernel(my_kernel, 0, pipeline_length, 0, 0, tile_width, tile_height, true)();

    for (uindex_t c = 0; c < tile_width; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            if (r < halo_radius) {
                REQUIRE(out_pipe_0::read());
            } else if (r < halo_radius + core_height) {
                REQUIRE(out_pipe_1::read());
            } else {
                REQUIRE(out_pipe_2::read());
            }
        }
    }

    REQUIRE(in_pipe_0::empty());
    REQUIRE(in_pipe_1::empty());
    REQUIRE(in_pipe_2::empty());
    REQUIRE(in_pipe_3::empty());
    REQUIRE(in_pipe_4::empty());
    REQUIRE(out_pipe_0::empty());
    REQUIRE(out_pipe_1::empty());
    REQUIRE(out_pipe_2::empty());
}

TEST_CASE("Incomplete Pipeline with i_generation != 0", "[tiling::ExecutionKernel]") {
    using Cell = uint8_t;
    auto trans_func = [](Stencil<Cell, 1> const &stencil) { return stencil[ID(0, 0)] + 1; };

    using in_pipe_0 = HostPipe<class IncompletePipelineInPipe0ID, Cell>;
    using in_pipe_1 = HostPipe<class IncompletePipelineInPipe1ID, Cell>;
    using in_pipe_2 = HostPipe<class IncompletePipelineInPipe2ID, Cell>;
    using in_pipe_3 = HostPipe<class IncompletePipelineInPipe3ID, Cell>;
    using in_pipe_4 = HostPipe<class IncompletePipelineInPipe4ID, Cell>;
    using out_pipe_0 = HostPipe<class IncompletePipelineOutPipe0ID, Cell>;
    using out_pipe_1 = HostPipe<class IncompletePipelineOutPipe1ID, Cell>;
    using out_pipe_2 = HostPipe<class IncompletePipelineOutPipe2ID, Cell>;
    using TestExecutionKernel =
        ExecutionKernel<decltype(trans_func), Cell, 1, 16, 64, 64, in_pipe_0, in_pipe_1, in_pipe_2,
        in_pipe_3, in_pipe_4, out_pipe_0, out_pipe_1, out_pipe_2>;

    for (int c = -16; c < 16 + 64; c++) {
        for (int r = -16; r < 16 + 64; r++) {
            if (r < 0) {
                in_pipe_0::write(0);
            } else if (r < 16) {
                in_pipe_1::write(0);
            } else if (r < 64 - 16) {
                in_pipe_2::write(0);
            } else if (r < 64) {
                in_pipe_3::write(0);
            } else {
                in_pipe_4::write(0);
            }
        }
    }

    TestExecutionKernel kernel(trans_func, 16, 20, 0, 0, 64, 64, 0);
    kernel.operator()();

    REQUIRE(in_pipe_0::empty());
    REQUIRE(in_pipe_1::empty());
    REQUIRE(in_pipe_2::empty());
    REQUIRE(in_pipe_3::empty());
    REQUIRE(in_pipe_4::empty());

    for (int c = 0; c < 64; c++) {
        for (int r = 0; r < 64; r++) {
            if (r < 16) {
                REQUIRE(out_pipe_0::read() == 4);
            } else if (r < 64 - 16) {
                REQUIRE(out_pipe_1::read() == 4);
            } else {
                REQUIRE(out_pipe_2::read() == 4);
            }
        }
    }

    REQUIRE(out_pipe_0::empty());
    REQUIRE(out_pipe_1::empty());
    REQUIRE(out_pipe_2::empty());
}