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
#include "../../IOPipeDebugging.hpp"
#include "../../TransFuncs.hpp"
#include <StencilStream/monotile/internal/StencilUpdateDesign.hpp>
#include <catch2/catch_all.hpp>

constexpr std::size_t max_grid_height = 256;
constexpr std::size_t max_grid_width = 128;

inline std::vector<sycl::queue> alloc_queues(std::size_t n_queues) {
    std::vector<sycl::queue> queues;
    sycl::device device(sycl::ext::intel::fpga_emulator_selector_v);
    for (std::size_t i = 0; i < n_queues; i++) {
        queues.push_back(sycl::queue(device));
    }
    return queues;
}

template <std::size_t temporal_parallelism, std::size_t spatial_parallelism, std::size_t n_kernels>
void test_monotile_stencil_update_design(std::size_t iteration_offset, std::size_t n_iterations) {
    using namespace stencil::internal;

    using Design = stencil::monotile::internal::StencilUpdateDesign<
        FPGATransFunc<1>, temporal_parallelism, spatial_parallelism, max_grid_height,
        max_grid_width, n_kernels, stencil::tdv::single_pass::InlineStrategy>;
    using CellVector = typename Design::CellVector;
    FPGATransFunc<1> trans_func;

    std::vector<sycl::queue> queues = alloc_queues(Design::n_kernels);
    Design design(trans_func, Cell::halo(), iteration_offset, n_iterations, queues);

    sycl::range<2> grid_range(max_grid_height, max_grid_width);

    sycl::queue queue(sycl::ext::intel::fpga_emulator_selector_v);
    queue.single_task([=]() {
        for (std::size_t r = 0; r < max_grid_height; r++) {
            for (std::size_t vect_c = 0; vect_c < int_ceil_div(max_grid_width, spatial_parallelism);
                 vect_c++) {
                CellVector cell_vector;
#pragma unroll
                for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                    cell_vector.value[i_cell] =
                        Cell{.r = int(r),
                             .c = int(vect_c * spatial_parallelism + i_cell),
                             .i_iteration = int(iteration_offset),
                             .i_subiteration = 0,
                             .status = CellStatus::Normal};
                }
                Design::work_in_pipe::write(cell_vector);
            }
        }
    });

    design.submit_work_kernels(iteration_offset, iteration_offset + n_iterations, grid_range);

    sycl::buffer<Cell, 2> out_buffer(grid_range);
    queue.submit([&](sycl::handler &cgh) {
        auto ac = sycl::accessor(out_buffer, cgh, sycl::write_only);
        cgh.single_task([=]() {
            for (std::size_t r = 0; r < max_grid_height; r++) {
                for (std::size_t vect_c = 0;
                     vect_c < int_ceil_div(max_grid_width, spatial_parallelism); vect_c++) {
                    CellVector cell_vector = Design::work_out_pipe::read();
#pragma unroll
                    for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                        ac[r][vect_c * spatial_parallelism + i_cell] = cell_vector.value[i_cell];
                    }
                }
            }
        });
    });

    sycl::host_accessor ac(out_buffer, sycl::read_only);
    for (std::size_t r = 0; r < max_grid_height; r++) {
        for (std::size_t c = 0; c < max_grid_width; c++) {
            Cell cell = ac[r][c];
            REQUIRE(cell.r == r);
            REQUIRE(cell.c == c);
            REQUIRE(cell.i_iteration == iteration_offset + n_iterations);
            REQUIRE(cell.i_subiteration == 0);
            REQUIRE(cell.status == CellStatus::Normal);
        }
    }
}

TEST_CASE("monotile::internal::StencilUpdateDesign", "[monotile::internal::StencilUpdateDesign]") {
    test_monotile_stencil_update_design<1, 1, 1>(0, 1);
    test_monotile_stencil_update_design<4, 1, 1>(0, 4);
    test_monotile_stencil_update_design<4, 1, 2>(0, 4);
    test_monotile_stencil_update_design<4, 2, 2>(0, 4);
}

template <std::size_t temporal_parallelism, std::size_t spatial_parallelism>
void test_monotile_local_stencil_update_design() {
    using Design = stencil::monotile::internal::LocalStencilUpdateDesign<
        FPGATransFunc<1>, temporal_parallelism, spatial_parallelism, max_grid_height,
        max_grid_width, 1, stencil::tdv::single_pass::InlineStrategy>;
    using CellVector = typename Design::CellVector;
    using Grid = stencil::monotile::Grid<Cell, spatial_parallelism>;
    using GridAccessor = typename Grid::template GridAccessor<sycl::access::mode::read_write>;

    Grid in_grid(max_grid_height, max_grid_width);
    Grid out_grid = in_grid.make_similar();
    {
        GridAccessor grid_ac(in_grid);
        for (std::size_t r = 0; r < max_grid_height; r++) {
            for (std::size_t c = 0; c < max_grid_width; c++) {
                grid_ac[r][c] = Cell{.r = int(r),
                                     .c = int(c),
                                     .i_iteration = 0,
                                     .i_subiteration = 0,
                                     .status = CellStatus::Normal};
            }
        }
    }

    std::vector<sycl::queue> queues = alloc_queues(Design::n_kernels);
    Design design(FPGATransFunc<1>(), Cell::halo(), 0, temporal_parallelism, queues);
    design.submit_pass(in_grid, out_grid, 0, temporal_parallelism);

    GridAccessor in_ac(in_grid);
    GridAccessor out_ac(out_grid);
    for (std::size_t r = 0; r < max_grid_height; r++) {
        for (std::size_t c = 0; c < max_grid_width; c++) {
            REQUIRE(in_ac[r][c].r == r);
            REQUIRE(in_ac[r][c].c == c);
            REQUIRE(in_ac[r][c].i_iteration == 0);
            REQUIRE(in_ac[r][c].i_subiteration == 0);
            REQUIRE(in_ac[r][c].status == CellStatus::Normal);
            REQUIRE(out_ac[r][c].r == r);
            REQUIRE(out_ac[r][c].c == c);
            REQUIRE(out_ac[r][c].i_iteration == temporal_parallelism);
            REQUIRE(out_ac[r][c].i_subiteration == 0);
            REQUIRE(out_ac[r][c].status == CellStatus::Normal);
        }
    }
}

TEST_CASE("monotile::internal::LocalStencilUpdateDesign",
          "[monotile::internal::LocalStencilUpdateDesign]") {
    test_monotile_local_stencil_update_design<1, 1>();
    test_monotile_local_stencil_update_design<4, 1>();
    test_monotile_local_stencil_update_design<1, 2>();
    test_monotile_local_stencil_update_design<4, 2>();
    test_monotile_local_stencil_update_design<1, 3>();
    test_monotile_local_stencil_update_design<4, 3>();
    test_monotile_local_stencil_update_design<1, 4>();
    test_monotile_local_stencil_update_design<4, 4>();
    test_monotile_local_stencil_update_design<1, 6>();
    test_monotile_local_stencil_update_design<4, 6>();
}

template <std::size_t temporal_parallelism, std::size_t spatial_parallelism>
void test_monotile_io_pipe_stencil_update_design(bool test_root) {
    using namespace stencil::internal;

    using Design = stencil::monotile::internal::IOPipeStencilUpdateDesign<
        FPGATransFunc<1>, temporal_parallelism, spatial_parallelism, max_grid_height,
        max_grid_width>;
    using CellVector = typename Design::CellVector;
    using Grid = typename Design::GridImpl;
    using GridAccessor = typename Grid::template GridAccessor<sycl::access::mode::read_write>;

    std::vector<sycl::queue> queues = alloc_queues(Design::n_kernels);
    Design design(FPGATransFunc<1>(), Cell::halo(), 0, 2 * temporal_parallelism, queues);

    Grid in_grid(max_grid_height, max_grid_width);
    Grid out_grid = in_grid.make_similar();
    if (test_root) {
        GridAccessor grid_ac(in_grid);
        for (std::size_t r = 0; r < max_grid_height; r++) {
            for (std::size_t c = 0; c < max_grid_width; c++) {
                grid_ac[r][c] = Cell{.r = int(r),
                                     .c = int(c),
                                     .i_iteration = 0,
                                     .i_subiteration = 0,
                                     .status = CellStatus::Normal};
            }
        }
    }
    std::size_t vect_grid_width = int_ceil_div(max_grid_width, spatial_parallelism);

    {
        IOPipeDebugManager &manager = IOPipeDebugManager::get_instance();
        InputPipeWriter lower_in_pipe = manager.get_input_pipe_writer(0);
        InputPipeWriter upper_in_pipe = manager.get_input_pipe_writer(1);

        for (std::size_t r = 0; r < max_grid_height; r++) {
            for (std::size_t vect_c = 0; vect_c < vect_grid_width; vect_c++) {
                CellVector vector;
                for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                    vector.value[i_cell] =
                        Cell{.r = int(r),
                             .c = int(vect_c * spatial_parallelism + i_cell),
                             .i_iteration = int(test_root ? 2 * int(temporal_parallelism)
                                                          : int(temporal_parallelism)),
                             .i_subiteration = 0,
                             .status = CellStatus::Normal};
                }

                auto pipeword = (pipeword_t *)&vector;
                constexpr std::size_t n_pipewords_per_vector = sizeof(CellVector) / pipeword_size;

                for (std::size_t i_pipeword = 0; i_pipeword < n_pipewords_per_vector;
                     i_pipeword++) {
                    std::size_t global_i =
                        (r * vect_grid_width + vect_c) * n_pipewords_per_vector + i_pipeword;
                    if (global_i % 2 == 0) {
                        lower_in_pipe.write<pipeword_t>(pipeword[i_pipeword]);
                    } else {
                        upper_in_pipe.write<pipeword_t>(pipeword[i_pipeword]);
                    }
                }
            }
        }
    }

    design.submit_pass(in_grid, out_grid, test_root ? 0 : temporal_parallelism,
                       2 * temporal_parallelism, test_root);
    design.wait_for_queues();

    if (test_root) {
        GridAccessor in_ac(in_grid);
        GridAccessor out_ac(out_grid);
        for (std::size_t r = 0; r < max_grid_height; r++) {
            for (std::size_t c = 0; c < max_grid_width; c++) {
                REQUIRE(in_ac[r][c].r == r);
                REQUIRE(in_ac[r][c].c == c);
                REQUIRE(in_ac[r][c].i_iteration == 0);
                REQUIRE(in_ac[r][c].i_subiteration == 0);
                REQUIRE(in_ac[r][c].status == CellStatus::Normal);
                REQUIRE(out_ac[r][c].r == r);
                REQUIRE(out_ac[r][c].c == c);
                REQUIRE(out_ac[r][c].i_iteration == 2 * temporal_parallelism);
                REQUIRE(out_ac[r][c].i_subiteration == 0);
                REQUIRE(out_ac[r][c].status == CellStatus::Normal);
            }
        }
    }

    {
        IOPipeDebugManager &manager = IOPipeDebugManager::get_instance();
        OutputPipeReader lower_out_pipe = manager.get_output_pipe_reader(2);
        OutputPipeReader upper_out_pipe = manager.get_output_pipe_reader(3);

        for (std::size_t r = 0; r < max_grid_height; r++) {
            for (std::size_t vect_c = 0; vect_c < vect_grid_width; vect_c++) {
                CellVector vector;
                auto pipeword = (pipeword_t *)&vector;
                constexpr std::size_t n_pipewords_per_vector = sizeof(CellVector) / pipeword_size;

                for (std::size_t i_pipeword = 0; i_pipeword < n_pipewords_per_vector;
                     i_pipeword++) {
                    std::size_t global_i =
                        (r * vect_grid_width + vect_c) * n_pipewords_per_vector + i_pipeword;
                    if (global_i % 2 == 0) {
                        pipeword[i_pipeword] = lower_out_pipe.read<pipeword_t>();
                    } else {
                        pipeword[i_pipeword] = upper_out_pipe.read<pipeword_t>();
                    }
                }

                for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                    std::size_t c = vect_c * spatial_parallelism + i_cell;
                    if (c < max_grid_width) {
                        Cell cell = vector.value[i_cell];
                        REQUIRE(cell.r == r);
                        REQUIRE(cell.c == c);
                        REQUIRE(cell.i_iteration ==
                                (test_root ? temporal_parallelism : 2 * temporal_parallelism));
                        REQUIRE(cell.i_subiteration == 0);
                        REQUIRE(cell.status == CellStatus::Normal);
                    }
                }
            }
        }
    }
}

TEST_CASE("monotile::internal::IOPipeStencilUpdateDesign",
          "[monotile::internal::IOPipeStencilUpdateDesign]") {
    test_monotile_io_pipe_stencil_update_design<4, 1>(true);
    test_monotile_io_pipe_stencil_update_design<4, 2>(true);
    test_monotile_io_pipe_stencil_update_design<4, 3>(true);
    test_monotile_io_pipe_stencil_update_design<4, 4>(true);
    test_monotile_io_pipe_stencil_update_design<4, 5>(true);
    test_monotile_io_pipe_stencil_update_design<4, 6>(true);

    test_monotile_io_pipe_stencil_update_design<4, 1>(false);
    test_monotile_io_pipe_stencil_update_design<4, 2>(false);
    test_monotile_io_pipe_stencil_update_design<4, 3>(false);
    test_monotile_io_pipe_stencil_update_design<4, 4>(false);
    test_monotile_io_pipe_stencil_update_design<4, 5>(false);
    test_monotile_io_pipe_stencil_update_design<4, 6>(false);
}