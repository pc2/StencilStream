/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
 * University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "../../TransFuncs.hpp"
#include <StencilStream/tiling/Grid.hpp>
#include <StencilStream/tiling/internal/HaloInjectionKernel.hpp>
#include <catch2/catch_all.hpp>

template <std::size_t spatial_parallelism, std::size_t max_tile_height, std::size_t max_tile_width,
          std::size_t halo_height, std::size_t halo_width>
void test_tiling_halo_injection_kernel(std::size_t grid_height, std::size_t grid_width) {
    using Grid = stencil::tiling::Grid<Cell, spatial_parallelism>;
    using CellVector = typename Grid::CellVector;
    using in_pipe = sycl::pipe<class halo_injection_in_pipe_id, CellVector>;
    using out_pipe = sycl::pipe<class halo_injection_out_pipe_id, CellVector>;

    using Kernel =
        stencil::tiling::internal::HaloInjectionKernel<Cell, spatial_parallelism, in_pipe, out_pipe,
                                                       max_tile_height, max_tile_width, halo_height,
                                                       halo_width>;

    sycl::queue queue;
    sycl::range<2> max_tile_range(max_tile_height, max_tile_width);
    sycl::range<2> halo_range(halo_height, halo_width);
    Grid grid(grid_height, grid_width);

    sycl::range<2> tile_id_range = grid.get_tile_id_range(max_tile_range);
    for (std::size_t tile_r = 0; tile_r < tile_id_range[0]; tile_r++) {
        for (std::size_t tile_c = 0; tile_c < tile_id_range[1]; tile_c++) {
            sycl::id<2> tile_id(tile_r, tile_c);

            sycl::event in_event = queue.submit([&](sycl::handler &cgh) {
                std::array<std::ptrdiff_t, 2> vect_tile_offset =
                    grid.get_haloed_tile_offset(tile_id, max_tile_range, halo_range, true, true);
                sycl::range<2> vect_tile_range =
                    grid.get_haloed_tile_range(tile_id, max_tile_range, halo_range, true, true);

                cgh.single_task([=]() {
                    for (std::size_t local_r = 0; local_r < vect_tile_range[0]; local_r++) {
                        for (std::size_t local_vect_c = 0; local_vect_c < vect_tile_range[1];
                             local_vect_c++) {
                            std::size_t r = local_r + vect_tile_offset[0];
                            std::size_t base_c =
                                (local_vect_c + vect_tile_offset[1]) * spatial_parallelism;
                            CellVector cell_vector;
#pragma unroll
                            for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                                cell_vector.value[i_cell] =
                                    Cell{int(r), int(base_c + i_cell), 0, 0, CellStatus::Normal};
                            }
                            in_pipe::write(cell_vector);
                        }
                    }
                });
            });

            sycl::event work_event = queue.single_task(Kernel(grid, tile_id, Cell::halo()));

            sycl::buffer<CellVector, 2> output(
                grid.get_haloed_tile_range(tile_id, max_tile_range, halo_range, true, false));
            sycl::event out_event = queue.submit([&](sycl::handler &cgh) {
                sycl::accessor ac(output, cgh, sycl::write_only);

                cgh.single_task([=]() {
                    for (std::size_t r = 0; r < ac.get_range()[0]; r++) {
                        for (std::size_t c = 0; c < ac.get_range()[1]; c++) {
                            ac[r][c] = out_pipe::read();
                        }
                    }
                });
            });

            sycl::host_accessor ac(output, sycl::read_only);
            std::array<std::ptrdiff_t, 2> vect_tile_offset =
                grid.get_haloed_tile_offset(tile_id, max_tile_range, halo_range, true, false);
            sycl::range<2> vect_tile_range =
                grid.get_haloed_tile_range(tile_id, max_tile_range, halo_range, true, false);
            for (std::size_t local_r = 0; local_r < vect_tile_range[0]; local_r++) {
                int r = local_r + vect_tile_offset[0];
                for (std::size_t local_vect_c = 0; local_vect_c < vect_tile_range[1];
                     local_vect_c++) {
                    CellVector vector = ac[local_r][local_vect_c];
                    int base_c = (local_vect_c + vect_tile_offset[1]) * spatial_parallelism;
                    for (int i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                        Cell cell = vector.value[i_cell];
                        int c = base_c + i_cell;
                        if (r >= 0 && c >= 0 && r < grid_height && c < grid_width) {
                            REQUIRE(cell.r == r);
                            REQUIRE(cell.c == c);
                            REQUIRE(cell.i_iteration == 0);
                            REQUIRE(cell.i_subiteration == 0);
                            REQUIRE(cell.status == CellStatus::Normal);
                        } else {
                            REQUIRE(cell.r == 0);
                            REQUIRE(cell.c == 0);
                            REQUIRE(cell.i_iteration == 0);
                            REQUIRE(cell.i_iteration == 0);
                            REQUIRE(cell.i_subiteration == 0);
                            REQUIRE(cell.status == CellStatus::Halo);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("tiling::HaloInjectionKernel", "[tiling::HaloInjectionKernel]") {
    test_tiling_halo_injection_kernel<1, 64, 64, 16, 16>(64, 64);
    test_tiling_halo_injection_kernel<1, 64, 64, 16, 16>(63, 63);
    test_tiling_halo_injection_kernel<1, 64, 64, 16, 16>(65, 65);
    test_tiling_halo_injection_kernel<1, 64, 64, 16, 16>(128, 128);
    test_tiling_halo_injection_kernel<1, 64, 64, 16, 16>(127, 127);
    test_tiling_halo_injection_kernel<1, 64, 64, 16, 16>(129, 129);
    test_tiling_halo_injection_kernel<1, 64, 64, 16, 16>(192, 192);
    test_tiling_halo_injection_kernel<1, 64, 64, 16, 16>(193, 193);

    test_tiling_halo_injection_kernel<2, 64, 64, 16, 32>(64, 64);
    test_tiling_halo_injection_kernel<2, 64, 64, 16, 32>(63, 63);
    test_tiling_halo_injection_kernel<2, 64, 64, 16, 32>(65, 65);
    test_tiling_halo_injection_kernel<2, 64, 64, 16, 32>(128, 128);
    test_tiling_halo_injection_kernel<2, 64, 64, 16, 32>(127, 127);
    test_tiling_halo_injection_kernel<2, 64, 64, 16, 32>(129, 129);
    test_tiling_halo_injection_kernel<2, 64, 64, 16, 32>(192, 192);
    test_tiling_halo_injection_kernel<2, 64, 64, 16, 32>(191, 191);
    test_tiling_halo_injection_kernel<2, 64, 64, 16, 32>(193, 193);
}