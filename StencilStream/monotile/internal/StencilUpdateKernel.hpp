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
namespace monotile {
namespace internal {

/**
 * \brief The execution kernel of the monotile architecture
 *
 * It receives the contents of a tile and it's halo from the `in_pipe`, applies the transition
 * function when applicable and writes the result to the `out_pipe`.
 *
 * With the monotile approach, the whole grid fits in one tile. This eliminates the need to
 * calculate the cells of the tile halo, reducing the cache size and number of loop iterations. More
 * is described in \ref monotile.
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
 * \tparam max_grid_height The maximum number of rows in the grid. This will define the size of the
 * column buffer.
 *
 * \tparam max_grid_width The maximum number of columns in the grid. This will define the bit width
 * of the cell indices.
 *
 * \tparam in_pipe The pipe to read from.
 *
 * \tparam out_pipe The pipe to write to.
 */
template <concepts::TransitionFunction TransFunc,
          tdv::single_pass::KernelArgument<TransFunc> TDVKernelArgument,
          std::size_t temporal_parallelism, std::size_t spatial_parallelism,
          std::size_t max_grid_height, std::size_t max_grid_width, typename in_pipe,
          typename out_pipe>
class StencilUpdateKernel {
  private:
    using Cell = typename TransFunc::Cell;
    using TDV = typename TransFunc::TimeDependentValue;
    using TDVLocalState = typename TDVKernelArgument::LocalState;
    using StencilImpl = Stencil<Cell, TransFunc::stencil_radius, TDV>;
    using CellVector = stencil::internal::Padded<std::array<Cell, spatial_parallelism>>;

    static constexpr std::size_t n_processing_elements =
        temporal_parallelism * TransFunc::n_subiterations;

    static constexpr std::size_t max_vect_grid_width =
        stencil::internal::int_ceil_div(max_grid_width, spatial_parallelism);

    // Round up the stencil buffer lead to the next integer multiple of the vector length.
    static constexpr std::size_t vect_stencil_buffer_lead =
        stencil::internal::int_ceil_div(TransFunc::stencil_radius, spatial_parallelism);
    static constexpr std::size_t stencil_buffer_lead =
        vect_stencil_buffer_lead * spatial_parallelism;
    static constexpr std::size_t stencil_buffer_height = 2 * TransFunc::stencil_radius + 1;
    static constexpr std::size_t stencil_buffer_width =
        TransFunc::stencil_radius + spatial_parallelism + stencil_buffer_lead;

    template <typename T> static constexpr T calc_pipeline_latency(T grid_width) {
        using namespace stencil::internal;
        T vect_grid_width = int_ceil_div<T>(grid_width, spatial_parallelism);
        return T(n_processing_elements * TransFunc::stencil_radius) * vect_grid_width +
               T(n_processing_elements * vect_stencil_buffer_lead);
    }

    template <typename T> static constexpr T calc_n_steps(T grid_height, T grid_width) {
        using namespace stencil::internal;
        T vect_grid_width = int_ceil_div<T>(grid_width, spatial_parallelism);
        return grid_height * vect_grid_width + calc_pipeline_latency<T>(grid_width);
    }

    using index_r_t = ac_int<std::bit_width(max_grid_height) + 1, true>;
    using uindex_r_t = ac_int<std::bit_width(max_grid_height), false>;

    using uindex_c_t = ac_int<std::bit_width(max_grid_width), false>;

    using index_vect_c_t = ac_int<std::bit_width(max_vect_grid_width) + 1, true>;
    using uindex_vect_c_t = ac_int<std::bit_width(max_vect_grid_width), false>;

    using uindex_step_t =
        ac_int<std::bit_width(calc_n_steps(max_grid_height, max_grid_width)), false>;

  public:
    /**
     * \brief Create and configure the execution kernel.
     *
     * \param trans_func The instance of the transition function to use.
     *
     * \param i_iteration The iteration index of the input cells.
     *
     * \param target_i_iteration The final, requested iteration index after the updates. This may be
     * higher than what the kernel can process in one pass. In this case, the kernel will compute
     * the maximum number of iterations.
     *
     * \param grid_height The number of cell rows in the grid.
     *
     * \param grid_width The number of cell columns in the grid.
     *
     * \param halo_value The value of cells outside the grid.
     *
     * \param tdv_kernel_argument The argument for the TDV system that is passed from the host to
     * the device. This may for example contain global memory accessors.
     */
    StencilUpdateKernel(TransFunc trans_func, std::size_t i_iteration,
                        std::size_t target_i_iteration, std::size_t grid_height,
                        std::size_t grid_width, Cell halo_value,
                        TDVKernelArgument tdv_kernel_argument)
        : trans_func(trans_func), i_iteration(i_iteration), target_i_iteration(target_i_iteration),
          grid_height(grid_height), grid_width(grid_width), halo_value(halo_value),
          tdv_kernel_argument(tdv_kernel_argument) {
        assert(TransFunc::stencil_radius <= grid_width && grid_width <= max_grid_width);
        assert(TransFunc::stencil_radius <= grid_height && grid_height <= max_grid_height);
        assert(grid_width >= 2);
        assert(grid_width <= max_grid_width);
        assert(grid_height <= max_grid_height);
    }

    /**
     * \brief Execute the kernel.
     */
    void operator()() const {
        using namespace stencil::internal;
        uindex_vect_c_t vect_grid_width = int_ceil_div<uindex_c_t>(grid_width, spatial_parallelism);

        [[intel::fpga_register]] index_r_t r[n_processing_elements];
        [[intel::fpga_register]] uindex_vect_c_t vect_c[n_processing_elements];
        TDVLocalState tdv_local_state(tdv_kernel_argument);

        // Initializing (output) column and row counters.
        index_r_t prev_r = 0;
        index_vect_c_t prev_vect_c = 0;
#pragma unroll
        for (std::size_t i = 0; i < n_processing_elements; i++) {
            prev_r -= TransFunc::stencil_radius;
            prev_vect_c -= vect_stencil_buffer_lead;
            if (prev_vect_c < 0) {
                prev_vect_c += vect_grid_width;
                prev_r -= 1;
            }
            r[i] = prev_r;
            vect_c[i] = prev_vect_c;
        }

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
            cache[2][max_vect_grid_width][std::bit_ceil(n_processing_elements)];
        [[intel::fpga_register]] Cell stencil_buffer[n_processing_elements][stencil_buffer_height]
                                                    [stencil_buffer_width];

        uindex_step_t n_steps = calc_n_steps(grid_height, grid_width);

        /*
         * OneAPI 2024.1 and newer finds a WAR memory dependency on the cache that it can't resolve
         * on its own. This issue is resolved by declaring that the distance between a read and a
         * write to the same memory location is at least two. This is ensured by requiring a minimal
         * grid width of two.
         */
        [[intel::ivdep(cache, 2)]] for (uindex_step_t i = 0; i < n_steps; i++) {
            CellVector carry;
            if (i < uindex_r_t(grid_height) * vect_grid_width) {
                carry = in_pipe::read();
            } else {
#pragma unroll
                for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                    carry.value[i_cell] = halo_value;
                }
            }

#pragma unroll
            for (std::size_t i_processing_element = 0; i_processing_element < n_processing_elements;
                 i_processing_element++) {
#pragma unroll
                for (std::size_t r = 0; r < stencil_buffer_height; r++) {
#pragma unroll
                    for (std::size_t c = 0; c < stencil_buffer_width - spatial_parallelism; c++) {
                        stencil_buffer[i_processing_element][r][c] =
                            stencil_buffer[i_processing_element][r][c + spatial_parallelism];
                    }
                }

                // Update the stencil buffer and cache with previous cache contents and the new
                // input cell.
                [[intel::fpga_register]] std::array<CellVector, stencil_buffer_height - 1>
                    in_cache_word = cache[r[i_processing_element][0]][vect_c[i_processing_element]]
                                         [i_processing_element];
                [[intel::fpga_register]] std::array<CellVector, stencil_buffer_height - 1>
                    out_cache_word;
#pragma unroll
                for (std::size_t cache_r = 0; cache_r < stencil_buffer_height; cache_r++) {
                    CellVector new_vect;
                    if (cache_r == stencil_buffer_height - 1) {
                        new_vect = carry;
                    } else {
                        new_vect = in_cache_word[cache_r];
                    }

#pragma unroll
                    for (std::size_t i_vector_cell = 0; i_vector_cell < spatial_parallelism;
                         i_vector_cell++) {
                        stencil_buffer[i_processing_element][cache_r]
                                      [TransFunc::stencil_radius + stencil_buffer_lead +
                                       i_vector_cell] = new_vect.value[i_vector_cell];
                    }

                    if (cache_r > 0) {
                        out_cache_word[cache_r - 1] = new_vect;
                    }
                }
                cache[(~r[i_processing_element])[0]][vect_c[i_processing_element]]
                     [i_processing_element] = out_cache_word;

                std::size_t pe_iteration = i_processing_element / TransFunc::n_subiterations;
                std::size_t n_iterations =
                    (i_iteration <= target_i_iteration)
                        ? std::min(target_i_iteration - i_iteration, temporal_parallelism)
                        : 0;

                if (pe_iteration < n_iterations) {
                    TDV tdv = tdv_local_state.get_time_dependent_value(pe_iteration);

                    bool v_halo_mask[stencil_buffer_height];
#pragma unroll
                    for (std::size_t mask_i = 0; mask_i < stencil_buffer_height; mask_i++) {
                        std::ptrdiff_t cell_r = std::ptrdiff_t(r[i_processing_element]) +
                                                std::ptrdiff_t(mask_i - TransFunc::stencil_radius);
                        v_halo_mask[mask_i] = cell_r >= 0 && cell_r < grid_height;
                    }

                    bool h_halo_mask[2 * TransFunc::stencil_radius + spatial_parallelism];
#pragma unroll
                    for (std::size_t mask_i = 0;
                         mask_i < 2 * TransFunc::stencil_radius + spatial_parallelism; mask_i++) {
                        std::ptrdiff_t c =
                            std::ptrdiff_t(vect_c[i_processing_element]) * spatial_parallelism -
                            TransFunc::stencil_radius + mask_i;
                        h_halo_mask[mask_i] = c >= 0 && c < grid_width;
                    }

#pragma unroll
                    for (std::size_t i_vector_cell = 0; i_vector_cell < spatial_parallelism;
                         i_vector_cell++) {
                        std::size_t c =
                            std::size_t(vect_c[i_processing_element]) * spatial_parallelism +
                            i_vector_cell;

                        StencilImpl stencil(sycl::id<2>(r[i_processing_element], c),
                                            sycl::range<2>(grid_height, grid_width),
                                            i_iteration + pe_iteration,
                                            i_processing_element % TransFunc::n_subiterations, tdv);

#pragma unroll
                        for (std::size_t cell_r = 0; cell_r < 2 * TransFunc::stencil_radius + 1;
                             cell_r++) {
#pragma unroll
                            for (std::size_t cell_c = 0; cell_c < 2 * TransFunc::stencil_radius + 1;
                                 cell_c++) {
                                if (v_halo_mask[cell_r] && h_halo_mask[cell_c + i_vector_cell]) {
                                    stencil[sycl::id<2>(cell_r, cell_c)] =
                                        stencil_buffer[i_processing_element][cell_r]
                                                      [cell_c + i_vector_cell];
                                } else {
                                    stencil[sycl::id<2>(cell_r, cell_c)] = halo_value;
                                }
                            }
                        }

                        carry.value[i_vector_cell] = trans_func(stencil);
                    }
                } else {
#pragma unroll
                    for (std::size_t i_vector_cell = 0; i_vector_cell < spatial_parallelism;
                         i_vector_cell++) {
                        carry.value[i_vector_cell] =
                            stencil_buffer[i_processing_element][TransFunc::stencil_radius]
                                          [TransFunc::stencil_radius + i_vector_cell];
                    }
                }

                vect_c[i_processing_element] += 1;
                if (vect_c[i_processing_element] == vect_grid_width) {
                    vect_c[i_processing_element] = 0;
                    r[i_processing_element] += 1;
                }
            }

            if (i >= calc_pipeline_latency<uindex_step_t>(grid_width)) {
                out_pipe::write(carry);
            }
        }
    }

  private:
    TransFunc trans_func;
    std::size_t i_iteration;
    std::size_t target_i_iteration;
    std::size_t grid_height;
    std::size_t grid_width;
    Cell halo_value;
    TDVKernelArgument tdv_kernel_argument;
};

} // namespace internal
} // namespace monotile
} // namespace stencil
