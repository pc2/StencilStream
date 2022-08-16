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
#pragma once
#include "../Concepts.hpp"
#include "../GenericID.hpp"
#include "../Helpers.hpp"
#include "../Index.hpp"
#include "../Padded.hpp"

#include <CL/sycl/accessor.hpp>
#include <optional>

namespace stencil {
namespace monotile {

/**
 * \brief A kernel that executes a stencil transition function using the monotile approach.
 *
 * It receives the contents of a tile and it's halo from the `in_pipe`, applies the transition
 * function when applicable and writes the result to the `out_pipe`.
 *
 * With the monotile approach, the whole grid fits in one tile. This eliminates the need to
 * calculate the cells of the tile halo, reducing the cache size and number of loop iterations. More
 * is described in \ref monotile.
 *
 * \tparam TransFunc The type of transition function to use.
 * \tparam n_processing_elements The number of processing elements to use. Similar to an unroll
 * factor for a loop.
 * \tparam output_tile_width The number of columns in a grid tile.
 * \tparam output_tile_height The number of rows in a grid tile.
 * \tparam in_pipe The pipe to read from.
 * \tparam out_pipe The pipe to write to.
 */
template <TransitionFunction TransFunc, tdv::GlobalState TDVGlobalState,
          uindex_t n_processing_elements, uindex_t tile_width, uindex_t tile_height,
          typename in_pipe, typename out_pipe>
requires(TransFunc::stencil_radius <= std::min(tile_width, tile_height)) &&
    (n_processing_elements % TransFunc::n_subgenerations == 0) class ExecutionKernel {
  public:
    using Cell = typename TransFunc::Cell;

    using TDVLocalState = typename TDVGlobalState::LocalState;
    using TDV = typename TDVLocalState::Value;
    static_assert(std::is_same<typename TransFunc::TimeDependentValue, TDV>());

    using StencilImpl = Stencil<Cell, TransFunc::stencil_radius, TDV>;

    /**
     * \brief The width and height of the stencil buffer.
     */
    static constexpr uindex_t stencil_diameter = StencilImpl::diameter;

    static constexpr uindex_t gens_per_pass = n_processing_elements / TransFunc::n_subgenerations;

    static constexpr uindex_t calc_pipeline_latency(uindex_t grid_height) {
        return n_processing_elements * TransFunc::stencil_radius * (grid_height + 1);
    }

    static constexpr uindex_t calc_n_iterations(uindex_t grid_width, uindex_t grid_height) {
        return grid_width * grid_height + calc_pipeline_latency(grid_height);
    }

    using index_stencil_t = typename StencilImpl::index_stencil_t;
    using uindex_stencil_t = typename StencilImpl::uindex_stencil_t;
    using StencilID = typename StencilImpl::StencilID;
    using StencilUID = typename StencilImpl::StencilUID;

    static constexpr unsigned long bits_1d = std::bit_width(std::max(tile_width, tile_height));
    using index_1d_t = ac_int<bits_1d + 1, true>;
    using uindex_1d_t = ac_int<bits_1d, false>;

    static constexpr unsigned long bits_2d = 2 * bits_1d;
    using index_2d_t = ac_int<bits_2d + 1, true>;
    using uindex_2d_t = ac_int<bits_2d, false>;

    static constexpr unsigned long bits_pes = std::bit_width(n_processing_elements);
    using index_pes_t = ac_int<bits_pes + 1, true>;
    using uindex_pes_t = ac_int<bits_pes, false>;

    static constexpr unsigned long bits_n_iterations =
        std::bit_width(calc_n_iterations(tile_width, tile_height));
    using index_n_iterations_t = ac_int<bits_n_iterations + 1, true>;
    using uindex_n_iterations_t = ac_int<bits_n_iterations, false>;

    /**
     * \brief Create and configure the execution kernel.
     *
     * \param trans_func The instance of the transition function to use.
     * \param i_generation The generation index of the input cells.
     * \param n_generations The number of generations to compute. If this number is bigger than
     * `n_processing_elements`, only `n_processing_elements` generations will be computed.
     * \param grid_width The number of cell columns in the grid.
     * \param grid_height The number of cell rows in the grid.
     * \param halo_value The value of cells outside the grid.
     */
    ExecutionKernel(TransFunc trans_func, uindex_t i_generation, uindex_t target_i_generation,
                    uindex_1d_t grid_width, uindex_1d_t grid_height, Cell halo_value,
                    TDVGlobalState global_state)
        : trans_func(trans_func), i_generation(i_generation),
          target_i_generation(target_i_generation), grid_width(grid_width),
          grid_height(grid_height), halo_value(halo_value), global_state(global_state) {}

    /**
     * \brief Execute the kernel.
     */
    void operator()() const {
        [[intel::fpga_register]] index_1d_t c[n_processing_elements];
        [[intel::fpga_register]] index_1d_t r[n_processing_elements];
        TDVLocalState local_state = global_state.build_local_state();

        // Initializing (output) column and row counters.
        index_1d_t prev_c = 0;
        index_1d_t prev_r = 0;
#pragma unroll
        for (uindex_pes_t i = 0; i < uindex_pes_t(n_processing_elements); i++) {
            c[i] = prev_c - TransFunc::stencil_radius;
            r[i] = prev_r - TransFunc::stencil_radius;
            if (r[i] < index_pes_t(0)) {
                r[i] += grid_height;
                c[i] -= 1;
            }
            prev_c = c[i];
            prev_r = r[i];
        }

        /*
         * The intel::numbanks attribute requires a power of two as it's argument and if the
         * number of processing elements isn't a power of two, it would produce an error. Therefore,
         * we calculate the next power of two and use it to allocate the cache. The compiler is
         * smart enough to see that these additional banks in the cache aren't used and therefore
         * optimizes them away.
         */
        [[intel::fpga_memory,
          intel::numbanks(2 * std::bit_ceil(n_processing_elements))]] Padded<Cell>
            cache[2][tile_height][std::bit_ceil(n_processing_elements)][stencil_diameter - 1];
        [[intel::fpga_register]] Cell stencil_buffer[n_processing_elements][stencil_diameter]
                                                    [stencil_diameter];

        uindex_n_iterations_t n_iterations = calc_n_iterations(grid_width, grid_height);
        for (uindex_n_iterations_t i = 0; i < n_iterations; i++) {
            Cell carry;
            if (i < uindex_n_iterations_t(grid_width * grid_height)) {
                carry = in_pipe::read();
            } else {
                carry = halo_value;
            }

#pragma unroll
            for (uindex_pes_t i_processing_element = 0;
                 i_processing_element < uindex_pes_t(n_processing_elements);
                 i_processing_element++) {
#pragma unroll
                for (uindex_stencil_t r = 0; r < uindex_stencil_t(stencil_diameter - 1); r++) {
#pragma unroll
                    for (uindex_stencil_t c = 0; c < uindex_stencil_t(stencil_diameter); c++) {
                        stencil_buffer[i_processing_element][c][r] =
                            stencil_buffer[i_processing_element][c][r + 1];
                    }
                }

                // Update the stencil buffer and cache with previous cache contents and the new
                // input cell.
#pragma unroll
                for (uindex_stencil_t cache_c = 0; cache_c < uindex_stencil_t(stencil_diameter);
                     cache_c++) {
                    Cell new_value;
                    if (cache_c == uindex_stencil_t(stencil_diameter - 1)) {
                        new_value = carry;
                    } else {
                        new_value = cache[c[i_processing_element][0]][r[i_processing_element]]
                                         [i_processing_element][cache_c]
                                             .value;
                    }

                    stencil_buffer[i_processing_element][cache_c][stencil_diameter - 1] = new_value;
                    if (cache_c > 0) {
                        cache[(~c[i_processing_element])[0]][r[i_processing_element]]
                             [i_processing_element][cache_c - 1]
                                 .value = new_value;
                    }
                }

                uindex_t pe_generation =
                    (i_generation + i_processing_element / TransFunc::n_subgenerations).to_uint();
                uindex_t pe_subgeneration =
                    (i_processing_element % TransFunc::n_subgenerations).to_uint();

                if (pe_generation < target_i_generation) {
                    StencilImpl stencil(
                        ID(c[i_processing_element], r[i_processing_element]),
                        UID(grid_width, grid_height), pe_generation, pe_subgeneration,
                        i_processing_element,
                        local_state.get_value(
                            (i_processing_element / TransFunc::n_subgenerations).to_uint()));

                    bool h_halo_mask[stencil_diameter];
                    bool v_halo_mask[stencil_diameter];
#pragma unroll
                    for (uindex_stencil_t mask_i = 0; mask_i < uindex_stencil_t(stencil_diameter);
                         mask_i++) {
                        // These computation assume that the central cell is in the grid. If it's
                        // not, the resulting value of this processing element will be discarded
                        // anyways, so this is safe.
                        if (mask_i < uindex_stencil_t(TransFunc::stencil_radius)) {
                            h_halo_mask[mask_i] = c[i_processing_element] >=
                                                  index_1d_t(TransFunc::stencil_radius - mask_i);
                            v_halo_mask[mask_i] = r[i_processing_element] >=
                                                  index_1d_t(TransFunc::stencil_radius - mask_i);
                        } else if (mask_i == uindex_stencil_t(TransFunc::stencil_radius)) {
                            h_halo_mask[mask_i] = true;
                            v_halo_mask[mask_i] = true;
                        } else {
                            h_halo_mask[mask_i] =
                                c[i_processing_element] <
                                grid_width + index_1d_t(TransFunc::stencil_radius - mask_i);
                            v_halo_mask[mask_i] =
                                r[i_processing_element] <
                                grid_height + index_1d_t(TransFunc::stencil_radius - mask_i);
                        }
                    }

#pragma unroll
                    for (uindex_stencil_t cell_c = 0; cell_c < uindex_stencil_t(stencil_diameter);
                         cell_c++) {
#pragma unroll
                        for (uindex_stencil_t cell_r = 0;
                             cell_r < uindex_stencil_t(stencil_diameter); cell_r++) {
                            if (h_halo_mask[cell_c] && v_halo_mask[cell_r]) {
                                stencil[StencilUID(cell_c, cell_r)] =
                                    stencil_buffer[i_processing_element][cell_c][cell_r];
                            } else {
                                stencil[StencilUID(cell_c, cell_r)] = halo_value;
                            }
                        }
                    }

                    carry = trans_func(stencil);
                } else {
                    carry = stencil_buffer[i_processing_element][TransFunc::stencil_radius]
                                          [TransFunc::stencil_radius];
                }

                r[i_processing_element] += 1;
                if (r[i_processing_element] == index_1d_t(grid_height)) {
                    r[i_processing_element] = 0;
                    c[i_processing_element] += 1;
                }
            }

            if (i >= uindex_n_iterations_t(calc_pipeline_latency(grid_height))) {
                out_pipe::write(carry);
            }
        }
    }

  private:
    TransFunc trans_func;
    uindex_t i_generation;
    uindex_t target_i_generation;
    uindex_1d_t grid_width;
    uindex_1d_t grid_height;
    Cell halo_value;
    TDVGlobalState global_state;
};

} // namespace monotile
} // namespace stencil