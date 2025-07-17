/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include <StencilStream/fpga_io/MemoryKernels.hpp>
#include <catch2/catch_all.hpp>

constexpr std::size_t max_grid_height = 64;
constexpr std::size_t max_grid_width = 32;

void test_complete_buffer_read_kernel(sycl::range<2> buffer_range) {
    using Cell = sycl::id<2>;
    using out_pipe = sycl::pipe<class complete_buffer_read_kernel_pipe_id, Cell>;
    using Kernel =
        stencil::fpga_io::CompleteBufferReadKernel<Cell, out_pipe, max_grid_height, max_grid_width>;

    sycl::buffer<Cell, 2> in_buffer(buffer_range);
    {
        sycl::host_accessor in_buffer_ac(in_buffer, sycl::write_only);
        for (std::size_t r = 0; r < buffer_range[0]; r++) {
            for (std::size_t c = 0; c < buffer_range[1]; c++) {
                in_buffer_ac[r][c] = sycl::id<2>(r, c);
            }
        }
    }

    sycl::queue queue;
    queue.submit([&](sycl::handler &cgh) { cgh.single_task(Kernel(in_buffer, cgh)); });

    sycl::buffer<Cell, 2> out_buffer(buffer_range);
    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor out_ac(out_buffer, cgh, sycl::write_only);

        cgh.single_task([=]() {
            for (std::size_t r = 0; r < buffer_range[0]; r++) {
                for (std::size_t c = 0; c < buffer_range[1]; c++) {
                    out_ac[r][c] = out_pipe::read();
                }
            }
        });
    });

    sycl::host_accessor out_buffer_ac(out_buffer, sycl::read_only);
    for (std::size_t r = 0; r < buffer_range[0]; r++) {
        for (std::size_t c = 0; c < buffer_range[1]; c++) {
            REQUIRE(out_buffer_ac[r][c] == sycl::id<2>(r, c));
        }
    }
}

TEST_CASE("fpga_io::CompleteBufferReadKernel", "[fpga_io]") {
    test_complete_buffer_read_kernel(sycl::range<2>(max_grid_height, max_grid_width));
    test_complete_buffer_read_kernel(sycl::range<2>(max_grid_height - 1, max_grid_width));
    test_complete_buffer_read_kernel(sycl::range<2>(max_grid_height, max_grid_width - 1));
    test_complete_buffer_read_kernel(sycl::range<2>(max_grid_height - 1, max_grid_width - 1));
}

void test_complete_buffer_write_kernel(sycl::range<2> buffer_range) {
    using Cell = sycl::id<2>;
    using in_pipe = sycl::pipe<class complete_buffer_write_kernel_pipe_id, Cell>;
    using Kernel =
        stencil::fpga_io::CompleteBufferWriteKernel<Cell, in_pipe, max_grid_height, max_grid_width>;

    sycl::queue queue;
    queue.single_task([=]() {
        for (std::size_t r = 0; r < buffer_range[0]; r++) {
            for (std::size_t c = 0; c < buffer_range[1]; c++) {
                in_pipe::write(sycl::id<2>(r, c));
            }
        }
    });

    sycl::buffer<Cell, 2> out_buffer(buffer_range);
    queue.submit([&](sycl::handler &cgh) { cgh.single_task(Kernel(out_buffer, cgh)); });

    sycl::host_accessor out_buffer_ac(out_buffer, sycl::read_only);
    for (std::size_t r = 0; r < buffer_range[0]; r++) {
        for (std::size_t c = 0; c < buffer_range[1]; c++) {
            REQUIRE(out_buffer_ac[r][c] == sycl::id<2>(r, c));
        }
    }
}

TEST_CASE("fpga_io::CompleteBufferWriteKernel", "[fpga_io]") {
    test_complete_buffer_write_kernel(sycl::range<2>(max_grid_height, max_grid_width));
    test_complete_buffer_write_kernel(sycl::range<2>(max_grid_height - 1, max_grid_width));
    test_complete_buffer_write_kernel(sycl::range<2>(max_grid_height, max_grid_width - 1));
    test_complete_buffer_write_kernel(sycl::range<2>(max_grid_height - 1, max_grid_width - 1));
}

void test_partial_buffer_read_kernel(sycl::range<2> buffer_range, sycl::id<2> offset,
                                     sycl::range<2> tile_range) {
    using Cell = sycl::id<2>;
    using out_pipe = sycl::pipe<class partial_buffer_read_kernel_pipe_id, Cell>;
    using Kernel =
        stencil::fpga_io::PartialBufferReadKernel<Cell, out_pipe, max_grid_height, max_grid_width>;

    sycl::buffer<Cell, 2> in_buffer(buffer_range);
    {
        sycl::host_accessor in_buffer_ac(in_buffer, sycl::write_only);
        for (std::size_t r = 0; r < buffer_range[0]; r++) {
            for (std::size_t c = 0; c < buffer_range[1]; c++) {
                in_buffer_ac[r][c] = sycl::id<2>(r, c);
            }
        }
    }

    sycl::queue queue;
    queue.submit(
        [&](sycl::handler &cgh) { cgh.single_task(Kernel(in_buffer, cgh, offset, tile_range)); });

    sycl::buffer<Cell, 2> out_buffer(tile_range);
    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor out_ac(out_buffer, cgh, sycl::write_only);

        cgh.single_task([=]() {
            for (std::size_t r = 0; r < tile_range[0]; r++) {
                for (std::size_t c = 0; c < tile_range[1]; c++) {
                    out_ac[r][c] = out_pipe::read();
                }
            }
        });
    });

    sycl::host_accessor out_buffer_ac(out_buffer, sycl::read_only);
    for (std::size_t r = 0; r < tile_range[0]; r++) {
        for (std::size_t c = 0; c < tile_range[1]; c++) {
            REQUIRE(out_buffer_ac[r][c] == sycl::id<2>(offset[0] + r, offset[1] + c));
        }
    }
}

TEST_CASE("fpga_io::PartialBufferReadKernel", "[fpga_io]") {
    test_partial_buffer_read_kernel(sycl::range<2>(max_grid_height, max_grid_width),
                                    sycl::id<2>(0, 0),
                                    sycl::range<2>(max_grid_height, max_grid_width));
    for (std::size_t tile_r = 0; tile_r < 2; tile_r++) {
        for (std::size_t tile_c = 0; tile_c < 2; tile_c++) {
            test_partial_buffer_read_kernel(
                sycl::range<2>(2 * max_grid_height, 2 * max_grid_width),
                sycl::id<2>(tile_r * max_grid_height, tile_c * max_grid_width),
                sycl::range<2>(max_grid_height, max_grid_width));
        }
    }
}

void test_partial_buffer_write_kernel(sycl::range<2> buffer_range, sycl::id<2> offset,
                                      sycl::range<2> tile_range) {
    using Cell = sycl::id<2>;
    using out_pipe = sycl::pipe<class partial_buffer_write_kernel_pipe_id, Cell>;
    using Kernel =
        stencil::fpga_io::PartialBufferWriteKernel<Cell, out_pipe, max_grid_height, max_grid_width>;

    sycl::buffer<Cell, 2> out_buffer(buffer_range);
    {
        sycl::host_accessor out_ac(out_buffer, sycl::write_only);
        for (std::size_t r = 0; r < buffer_range[0]; r++) {
            for (std::size_t c = 0; c < buffer_range[1]; c++) {
                out_ac[r][c] = sycl::id<2>(1, 1);
            }
        }
    }

    sycl::queue queue;
    queue.single_task([=]() {
        for (std::size_t r = 0; r < tile_range[0]; r++) {
            for (std::size_t c = 0; c < tile_range[1]; c++) {
                out_pipe::write(sycl::id<2>(r + offset[0], c + offset[1]));
            }
        }
    });

    queue.submit(
        [&](sycl::handler &cgh) { cgh.single_task(Kernel(out_buffer, cgh, offset, tile_range)); });

    sycl::host_accessor out_buffer_ac(out_buffer, sycl::read_only);
    for (std::size_t r = 0; r < buffer_range[0]; r++) {
        for (std::size_t c = 0; c < buffer_range[1]; c++) {
            bool within_tile = r >= offset[0] && r < offset[0] + tile_range[0];
            within_tile &= c >= offset[1] && c < offset[1] + tile_range[1];
            if (within_tile) {
                REQUIRE(out_buffer_ac[r][c] == sycl::id<2>(r, c));
            } else {
                REQUIRE(out_buffer_ac[r][c] == sycl::id<2>(1, 1));
            }
        }
    }
}

TEST_CASE("fpga_io::PartialBufferWriteKernel", "[fpga_io]") {
    test_partial_buffer_write_kernel(sycl::range<2>(max_grid_height, max_grid_width),
                                     sycl::id<2>(0, 0),
                                     sycl::range<2>(max_grid_height, max_grid_width));
    for (std::size_t tile_r = 0; tile_r < 2; tile_r++) {
        for (std::size_t tile_c = 0; tile_c < 2; tile_c++) {
            test_partial_buffer_write_kernel(
                sycl::range<2>(2 * max_grid_height, 2 * max_grid_width),
                sycl::id<2>(tile_r * max_grid_height, tile_c * max_grid_width),
                sycl::range<2>(max_grid_height, max_grid_width));
        }
    }
}
