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
#include <StencilStream/internal/DualIOPipeKernels.hpp>
#include <catch2/catch_all.hpp>
#include <filesystem>
#include <fstream>
#include <random>

static std::size_t read_bytes_from_file_1 = 0;
static std::size_t read_bytes_from_file_3 = 0;

template <std::size_t vector_length> void test_dual_io_pipe_recv_kernel(std::size_t n_cells) {
    using namespace stencil::internal;
    struct Cell {
        std::size_t i;
        char padding[32 - sizeof(std::size_t)];
    };
    using Vect = std::array<Cell, vector_length>;
    using recv_pipe = sycl::pipe<class recv_pipe_id, Vect, 512>;
    using Kernel = DualIOPipeRecvKernel<Vect, kernel_input_ch0, kernel_input_ch1, recv_pipe>;
    std::size_t n_vectors = int_ceil_div(n_cells, vector_length);

    std::uniform_int_distribution<std::size_t> seed_distribution(
        0, std::numeric_limits<std::size_t>::max() - n_cells);
    std::random_device rd;
    std::size_t seed = seed_distribution(rd);

    {
        std::fstream lower_out_file("0", std::ios_base::out | std::ios_base::app |
                                             std::ios_base::binary);
        std::fstream upper_out_file("2", std::ios_base::out | std::ios_base::app |
                                             std::ios_base::binary);

        for (std::size_t i_vector = 0; i_vector < n_vectors; i_vector++) {
            for (std::size_t i_cell = 0; i_cell < vector_length; i_cell++) {
                std::size_t i = i_vector * vector_length + i_cell;
                Cell cell{seed + i};
                if (i % 2 == 0) {
                    lower_out_file.write((char *)&cell, sizeof(Cell));
                } else {
                    upper_out_file.write((char *)&cell, sizeof(Cell));
                }
            }
        }
    }

    sycl::queue queue;
    queue.single_task(Kernel(n_vectors));

    sycl::buffer<Cell, 1> out_buffer = sycl::range<1>(n_cells);
    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor ac(out_buffer, cgh, sycl::write_only);

        cgh.single_task([=]() {
            for (std::size_t i_vector = 0; i_vector < n_vectors; i_vector++) {
                Vect vect = recv_pipe::read();
#pragma unroll
                for (std::size_t i_cell = 0; i_cell < vector_length; i_cell++) {
                    std::size_t i = i_vector * vector_length + i_cell;
                    if (i < n_cells) {
                        ac[i] = vect[i_cell];
                    }
                }
            }
        });
    });

    sycl::host_accessor ac(out_buffer, sycl::read_only);
    for (size_t i = 0; i < n_cells; i++) {
        REQUIRE(ac[i].i == seed + i);
    }
}

TEST_CASE("internal::DualIOPipeRecvKernel", "[DualIOPipeKernels]") {
    // Power-of-two cell counts
    test_dual_io_pipe_recv_kernel<1>(32 * 1024);
    test_dual_io_pipe_recv_kernel<2>(32 * 1024);
    test_dual_io_pipe_recv_kernel<4>(32 * 1024);

    // Non-power-of-two cell counts
    test_dual_io_pipe_recv_kernel<1>(127 * 127);
    test_dual_io_pipe_recv_kernel<2>(127 * 127);
    test_dual_io_pipe_recv_kernel<4>(127 * 127);
}

template <std::size_t vector_length> void test_dual_io_pipe_send_kernel(std::size_t n_cells) {
    using namespace stencil::internal;

    struct Cell {
        std::size_t i;
        char padding[32 - sizeof(std::size_t)];
    };
    using Vect = std::array<Cell, vector_length>;
    using send_pipe = sycl::pipe<class send_pipe_id, Vect, 512>;
    using Kernel = DualIOPipeSendKernel<Vect, kernel_output_ch0, kernel_output_ch1, send_pipe>;
    std::size_t n_vectors = int_ceil_div(n_cells, vector_length);

    std::uniform_int_distribution<std::size_t> seed_distribution(
        0, std::numeric_limits<std::size_t>::max() - n_cells);
    std::random_device rd;
    std::size_t seed = seed_distribution(rd);

    sycl::queue queue;

    queue.single_task([=]() {
        for (std::size_t i_vector = 0; i_vector < n_vectors; i_vector++) {
            Vect vect;
#pragma unroll
            for (std::size_t i_cell = 0; i_cell < vector_length; i_cell++) {
                std::size_t i = i_vector * vector_length + i_cell;
                vect[i_cell] = Cell{seed + i};
            }
            send_pipe::write(vect);
        }
    });
    queue.single_task(Kernel(n_vectors));
    queue.wait();

    {
        std::fstream lower_out_file("1", std::ios_base::in | std::ios_base::binary);
        lower_out_file.seekg(read_bytes_from_file_1);
        std::fstream upper_out_file("3", std::ios_base::in | std::ios_base::binary);
        upper_out_file.seekg(read_bytes_from_file_3);

        for (std::size_t i_vector = 0; i_vector < n_vectors; i_vector++) {
            for (std::size_t i_cell = 0; i_cell < vector_length; i_cell++) {
                Cell cell;
                std::size_t i = i_vector * vector_length + i_cell;
                if (i % 2 == 0) {
                    lower_out_file.read((char *)&cell, sizeof(Cell));
                    REQUIRE(lower_out_file.gcount() == sizeof(Cell));
                    read_bytes_from_file_1 += sizeof(Cell);
                } else {
                    upper_out_file.read((char *)&cell, sizeof(Cell));
                    REQUIRE(upper_out_file.gcount() == sizeof(Cell));
                    read_bytes_from_file_3 += sizeof(Cell);
                }
                REQUIRE(cell.i == seed + i);
            }
        }
    }
}

TEST_CASE("internal::DualIOPipeSendKernel", "[DualIOPipeKernels]") {
    // Power-of-two cell counts
    test_dual_io_pipe_send_kernel<1>(32 * 1024);
    test_dual_io_pipe_send_kernel<2>(32 * 1024);
    test_dual_io_pipe_send_kernel<4>(32 * 1024);

    // Non-power-of-two cell counts
    test_dual_io_pipe_send_kernel<1>(127 * 127);
    test_dual_io_pipe_send_kernel<2>(127 * 127);
    test_dual_io_pipe_send_kernel<4>(127 * 127);
}