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
#pragma once
#include "../../Concepts.hpp"
#include "../../internal/Helpers.hpp"
#include "../../tdv/SinglePassStrategies.hpp"
#include <sycl/ext/intel/ac_types/ac_int.hpp>

namespace stencil {
namespace tiling {
namespace internal {

/**
 * \brief A kernel that executes a stencil transition function on a tile.
 *
 * It receives the contents of a tile and it's halo from the `in_pipe`, applies the transition
 * function when applicable and writes the result to the `out_pipe`.
 *
 * \tparam TransFunc The type of transition function to use.
 *
 * \tparam TDVKernelArgument The type of parameter for the TDV system that is passed from the host
 * to the kernel.
 *
 * \tparam temporal_parallelism The number of iterations to compute in parallel. Notice that
 * subiterations within one iteration are always computed in parallel.
 *
 * \tparam spatial_parallelsim The number of cells to update in parallel within one iteration.
 *
 * \tparam output_tile_height The number of rows in a grid tile.
 *
 * \tparam output_tile_width The number of columns in a grid tile.
 *
 * \tparam in_pipe The pipe to read from.
 *
 * \tparam out_pipe The pipe to write to.
 */
template <concepts::TransitionFunction TransFunc,
          tdv::single_pass::KernelArgument<TransFunc> TDVKernelArgument,
          std::size_t local_temporal_parallelism, std::size_t remaining_temporal_parallelism,
          std::size_t spatial_parallelism, std::size_t output_tile_height,
          std::size_t output_tile_width, typename in_pipe, typename out_pipe>
class StencilUpdateKernel {
  private:
    using Cell = typename TransFunc::Cell;
    using CellVector = stencil::internal::Padded<std::array<Cell, spatial_parallelism>>;
    using TDV = typename TransFunc::TimeDependentValue;
    using StencilImpl = Stencil<Cell, TransFunc::stencil_radius, TDV>;
    using TDVLocalState = typename TDVKernelArgument::LocalState;

    static constexpr std::size_t stencil_radius = TransFunc::stencil_radius;
    static constexpr std::size_t vect_stencil_buffer_lead =
        stencil::internal::int_ceil_div(stencil_radius, spatial_parallelism);
    static constexpr std::size_t stencil_buffer_lead =
        vect_stencil_buffer_lead * spatial_parallelism;
    static constexpr std::size_t stencil_buffer_height = 2 * stencil_radius + 1;
    static constexpr std::size_t stencil_buffer_width =
        stencil_radius + spatial_parallelism + stencil_buffer_lead;

    static constexpr std::size_t n_processing_elements =
        local_temporal_parallelism * TransFunc::n_subiterations;

    static constexpr std::size_t halo_height =
        stencil_radius * (local_temporal_parallelism + remaining_temporal_parallelism) *
        TransFunc::n_subiterations;
    static constexpr std::size_t halo_width =
        stencil_buffer_lead * (local_temporal_parallelism + remaining_temporal_parallelism) *
        TransFunc::n_subiterations;
    static constexpr std::size_t vect_halo_width = halo_width / spatial_parallelism;

    static constexpr std::size_t local_halo_height =
        stencil_radius * local_temporal_parallelism * TransFunc::n_subiterations;
    static constexpr std::size_t local_halo_width =
        stencil_buffer_lead * local_temporal_parallelism * TransFunc::n_subiterations;
    static constexpr std::size_t vect_local_halo_width = local_halo_width / spatial_parallelism;

    static constexpr std::size_t vect_output_tile_width = output_tile_width / spatial_parallelism;
    static_assert(output_tile_width % spatial_parallelism == 0);

    static constexpr std::size_t input_tile_height = 2 * halo_height + output_tile_height;
    static constexpr std::size_t vect_input_tile_width =
        2 * vect_halo_width + vect_output_tile_width;
    static constexpr std::size_t input_tile_width = vect_input_tile_width * spatial_parallelism;

    static constexpr unsigned long bits_1d =
        std::bit_width(std::max(input_tile_height, input_tile_width));
    using index_1d_t = ac_int<bits_1d + 1, true>;
    using uindex_1d_t = ac_int<bits_1d, false>;

  public:
    /**
     * \brief Create and configure the execution kernel.
     *
     * \param trans_func The instance of the transition function to use.
     *
     * \param i_iteration The iteration index of the input cells.
     *
     * \param target_i_iteration The number of iterations to compute. If this number is bigger
     * than `n_processing_elements`, only `n_processing_elements` iterations will be computed.
     *
     * \param grid_r_offset The row offset of the processed tile relative to the grid's origin,
     * not including the halo. For example, for the most north-western tile the offset will always
     * be (0,0), not (-halo_radius,-halo_radius)
     *
     * \param grid_c_offset The column offset of the processed tile relative to the grid's origin.
     * See `grid_r_offset` for details.
     *
     * \param grid_height The number of cell rows in the grid.
     *
     * \param grid_width The number of cell columns in the grid.
     *
     * \param halo_value The value of cells in the grid halo.
     *
     * \param tdv_kernel_argument The argument for the TDV system that is passed from the host to
     * the device. This may for example contain global memory accessors.
     */
    StencilUpdateKernel(TransFunc trans_func, std::size_t i_iteration,
                        std::size_t target_i_iteration, std::size_t grid_r_offset,
                        std::size_t grid_c_offset, std::size_t grid_height, std::size_t grid_width,
                        Cell halo_value, TDVKernelArgument tdv_kernel_argument)
        : trans_func(trans_func), i_iteration(i_iteration), target_i_iteration(target_i_iteration),
          grid_r_offset(grid_r_offset), grid_c_offset(grid_c_offset), grid_height(grid_height),
          grid_width(grid_width), halo_value(halo_value), tdv_kernel_argument(tdv_kernel_argument) {
        assert(grid_r_offset % output_tile_height == 0);
        assert(grid_c_offset % output_tile_width == 0);
    }

    static constexpr std::size_t get_halo_height() { return halo_height; }

    static constexpr std::size_t get_halo_width() { return halo_width; }

    static constexpr std::size_t get_vect_halo_width() { return vect_halo_width; }

    /**
     * \brief Execute the configured operations.
     */
    void operator()() const {
        using namespace stencil::internal;

        TDVLocalState tdv_local_state(tdv_kernel_argument);

        /*
         * The intel::numbanks attribute requires a power of two as it's argument and if the
         * number of processing elements isn't a power of two, it would produce an error. Therefore,
         * we calculate the next power of two and use it to allocate the cache. The compiler is
         * smart enough to see that these additional banks in the cache aren't used and therefore
         * optimizes them away.
         */
        [[intel::fpga_memory,
          intel::numbanks(
              2 * std::bit_ceil(
                      n_processing_elements))]] std::array<CellVector, stencil_buffer_height - 1>
            cache[2][vect_input_tile_width][std::bit_ceil(n_processing_elements)];
        [[intel::fpga_register]] Cell stencil_buffer[n_processing_elements][stencil_buffer_height]
                                                    [stencil_buffer_width];

        uindex_1d_t output_tile_section_height =
            std::min(output_tile_height, grid_height - grid_r_offset);
        uindex_1d_t output_tile_section_width =
            std::min(output_tile_width, grid_width - grid_c_offset);
        uindex_1d_t vect_output_tile_section_width =
            int_ceil_div<uindex_1d_t>(output_tile_section_width, spatial_parallelism);

        uindex_1d_t input_tile_section_height = output_tile_section_height + 2 * halo_height;
        uindex_1d_t vect_input_tile_section_width =
            vect_output_tile_section_width + 2 * vect_halo_width;

        /*
         * OneAPI 2024.1 and newer finds a WAR memory dependency on the cache that it can't resolve
         * on its own. For this, we have to declare that the distance between a read and a write is
         * at least two iterations. For the tiling architecture, this is always the case since the
         * minimal input tile section width is three (1 left halo column, 1 output tile column, 1
         * right halo column). Thus, using the ivdep attribute is safe here.
         */
        [[intel::loop_coalesce(2)]] for (index_1d_t input_tile_r = 0;
                                         input_tile_r < input_tile_section_height; input_tile_r++) {
            [[intel::ivdep(cache, 2)]] for (index_1d_t vect_input_tile_c = 0;
                                            vect_input_tile_c < vect_input_tile_section_width;
                                            vect_input_tile_c++) {
                [[intel::fpga_register]] CellVector carry = in_pipe::read();

#pragma unroll
                for (std::size_t i_processing_element = 0;
                     i_processing_element < n_processing_elements; i_processing_element++) {

                    index_1d_t rel_input_grid_r =
                        index_1d_t(input_tile_r) - index_1d_t(halo_height) -
                        index_1d_t(i_processing_element * TransFunc::stencil_radius);
                    std::size_t input_grid_r = grid_r_offset + std::ptrdiff_t(rel_input_grid_r);

                    uindex_1d_t input_tile_c = vect_input_tile_c * spatial_parallelism;
                    index_1d_t rel_input_grid_c =
                        index_1d_t(input_tile_c) - index_1d_t(halo_width) -
                        index_1d_t(i_processing_element * stencil_buffer_lead);
                    std::size_t input_grid_c = grid_c_offset + std::ptrdiff_t(rel_input_grid_c);

                    /*
                     * Shift every value in the stencil_buffer left.
                     * This operation does not touch the values in the right-most column, which will
                     * be filled from the cache and the new input value later.
                     */
#pragma unroll
                    for (std::size_t r = 0; r < stencil_buffer_height; r++) {
#pragma unroll
                        for (std::size_t c = 0; c < stencil_buffer_width - spatial_parallelism;
                             c++) {
                            stencil_buffer[i_processing_element][r][c] =
                                stencil_buffer[i_processing_element][r][c + spatial_parallelism];
                        }
                    }

                    // Update the stencil buffer and cache with previous cache contents and the new
                    // input cell.
                    [[intel::fpga_register]] std::array<CellVector, stencil_buffer_height - 1>
                        in_cache_word =
                            cache[input_tile_r[0]][vect_input_tile_c][i_processing_element];
                    [[intel::fpga_register]] std::array<CellVector, stencil_buffer_height - 1>
                        out_cache_word;
#pragma unroll
                    for (std::size_t cache_r = 0; cache_r < stencil_buffer_height; cache_r++) {
                        CellVector new_vector;
                        if (cache_r == stencil_buffer_height - 1) {
                            // grid_*_offset is unsigned, can not directly test for negativity.
                            bool is_halo_row = (grid_r_offset == 0 && rel_input_grid_r < 0) ||
                                               input_grid_r >= grid_height;

#pragma unroll
                            for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                                bool is_halo_cell =
                                    is_halo_row ||
                                    (grid_c_offset == 0 && (rel_input_grid_c + i_cell) < 0) ||
                                    (input_grid_c + i_cell >= grid_width);
                                if (is_halo_cell) {
                                    new_vector.value[i_cell] = halo_value;
                                } else {
                                    new_vector.value[i_cell] = carry.value[i_cell];
                                }
                            }
                        } else {
                            new_vector = in_cache_word[cache_r];
                        }

#pragma unroll
                        for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                            stencil_buffer[i_processing_element][cache_r]
                                          [stencil_buffer_width - spatial_parallelism + i_cell] =
                                              new_vector.value[i_cell];
                        }

                        if (cache_r > 0) {
                            out_cache_word[cache_r - 1] = new_vector;
                        }
                    }
                    cache[(~input_tile_r)[0]][vect_input_tile_c][i_processing_element] =
                        out_cache_word;

                    std::size_t pe_iteration =
                        i_iteration +
                        std::size_t(i_processing_element / TransFunc::n_subiterations);
                    std::size_t pe_subiteration = i_processing_element % TransFunc::n_subiterations;
                    std::size_t output_grid_r = input_grid_r - TransFunc::stencil_radius;
                    TDV tdv = tdv_local_state.get_time_dependent_value(i_processing_element /
                                                                       TransFunc::n_subiterations);

#pragma unroll
                    for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                        std::size_t output_grid_c = input_grid_c - stencil_buffer_lead + i_cell;
                        StencilImpl stencil(sycl::id<2>(output_grid_r, output_grid_c),
                                            sycl::range<2>(grid_height, grid_width), pe_iteration,
                                            pe_subiteration, tdv);

#pragma unroll
                        for (std::size_t stencil_r = 0; stencil_r < 2 * stencil_radius + 1;
                             stencil_r++) {
#pragma unroll
                            for (std::size_t stencil_c = 0; stencil_c < 2 * stencil_radius + 1;
                                 stencil_c++) {
                                stencil[sycl::id<2>(stencil_r, stencil_c)] =
                                    stencil_buffer[i_processing_element][stencil_r]
                                                  [i_cell + stencil_c];
                            }
                        }

                        if (output_grid_c < grid_width) {
                            if (pe_iteration < target_i_iteration) {
                                carry.value[i_cell] = trans_func(stencil);
                            } else {
                                carry.value[i_cell] =
                                    stencil_buffer[i_processing_element][stencil_radius]
                                                  [stencil_radius + i_cell];
                            }
                        } else {
                            carry.value[i_cell] = halo_value;
                        }
                    }
                }

                bool is_valid_output =
                    (input_tile_r >= uindex_1d_t(2 * local_halo_height)) &&
                    (vect_input_tile_c >= uindex_1d_t(2 * vect_local_halo_width));

                if (is_valid_output) {
                    out_pipe::write(carry);
                }
            }
        }
    }

  private:
    TransFunc trans_func;
    std::size_t i_iteration;
    std::size_t target_i_iteration;
    std::size_t grid_r_offset;
    std::size_t grid_c_offset;
    std::size_t grid_height;
    std::size_t grid_width;
    Cell halo_value;
    TDVKernelArgument tdv_kernel_argument;
};

} // namespace internal
} // namespace tiling
} // namespace stencil