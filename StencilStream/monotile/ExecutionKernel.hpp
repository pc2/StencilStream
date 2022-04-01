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
#include "../GenericID.hpp"
#include "../Helpers.hpp"
#include "../Index.hpp"
#include "../Padded.hpp"
#include "../Stencil.hpp"
#include <optional>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

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
 * \tparam T Cell value type.
 * \tparam stencil_radius The static, maximal Chebyshev distance of cells in a stencil to the
 * central cell \tparam pipeline_length The number of pipeline stages to use. Similar to an unroll
 * factor for a loop. \tparam output_tile_width The number of columns in a grid tile. \tparam
 * output_tile_height The number of rows in a grid tile. \tparam in_pipe The pipe to read from.
 * \tparam out_pipe The pipe to write to.
 */
template <typename TransFunc, typename T, uindex_t stencil_radius, uindex_t pipeline_length,
          uindex_t tile_width, uindex_t tile_height, typename in_pipe, typename out_pipe>
class ExecutionKernel {
  public:
    static_assert(
        std::is_invocable_r<T, TransFunc const, Stencil<T, stencil_radius> const &>::value);
    static_assert(stencil_radius >= 1);
    static_assert(stencil_radius <= std::min(tile_width, tile_height));

    using StencilImpl = Stencil<T, stencil_radius>;

    /**
     * \brief The width and height of the stencil buffer.
     */
    static constexpr uindex_t stencil_diameter = Stencil<T, stencil_radius>::diameter;

    static constexpr uindex_t calc_pipeline_latency(uindex_t grid_height) {
        return pipeline_length * stencil_radius * (grid_height + 1);
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

    static constexpr unsigned long bits_plen = std::bit_width(pipeline_length);
    using index_plen_t = ac_int<bits_plen + 1, true>;
    using uindex_plen_t = ac_int<bits_plen, false>;

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
     * `pipeline_length`, only `pipeline_length` generations will be computed.
     * \param grid_width The number of cell columns in the grid.
     * \param grid_height The number of cell rows in the grid.
     * \param halo_value The value of cells outside the grid.
     */
    ExecutionKernel(TransFunc trans_func, uindex_t i_generation, uindex_t n_generations,
                    uindex_1d_t grid_width, uindex_1d_t grid_height, T halo_value)
        : trans_func(trans_func), i_generation(i_generation), n_generations(n_generations),
          grid_width(grid_width), grid_height(grid_height), halo_value(halo_value) {}

    /**
     * \brief Execute the kernel.
     */
    void operator()() const {
        [[intel::fpga_register]] index_1d_t c[pipeline_length];
        [[intel::fpga_register]] index_1d_t r[pipeline_length];

        // Initializing (output) column and row counters.
        index_1d_t prev_c = 0;
        index_1d_t prev_r = 0;
#pragma unroll
        for (uindex_plen_t i = 0; i < uindex_plen_t(pipeline_length); i++) {
            c[i] = prev_c - stencil_radius;
            r[i] = prev_r - stencil_radius;
            if (r[i] < index_plen_t(0)) {
                r[i] += grid_height;
                c[i] -= 1;
            }
            prev_c = c[i];
            prev_r = r[i];
        }

        /*
         * The intel::numbanks attribute requires a power of two as it's argument and if the
         * pipeline length isn't a power of two, it would produce an error. Therefore, we calculate
         * the next power of two and use it to allocate the cache. The compiler is smart enough to
         * see that these additional banks in the cache aren't used and therefore optimizes them
         * away.
         */
        [[intel::fpga_memory, intel::numbanks(2 * std::bit_ceil(pipeline_length))]] Padded<T>
            cache[2][tile_height][std::bit_ceil(pipeline_length)][stencil_diameter - 1];
        [[intel::fpga_register]] T stencil_buffer[pipeline_length][stencil_diameter]
                                                 [stencil_diameter];

        uindex_n_iterations_t n_iterations = calc_n_iterations(grid_width, grid_height);
        for (uindex_n_iterations_t i = 0; i < n_iterations; i++) {
            T carry;
            if (i < uindex_n_iterations_t(grid_width * grid_height)) {
                carry = in_pipe::read();
            } else {
                carry = halo_value;
            }

#pragma unroll
            for (uindex_plen_t stage = 0; stage < uindex_plen_t(pipeline_length); stage++) {
#pragma unroll
                for (uindex_stencil_t r = 0; r < uindex_stencil_t(stencil_diameter - 1); r++) {
#pragma unroll
                    for (uindex_stencil_t c = 0; c < uindex_stencil_t(stencil_diameter); c++) {
                        stencil_buffer[stage][c][r] = stencil_buffer[stage][c][r + 1];
                    }
                }

                // Update the stencil buffer and cache with previous cache contents and the new
                // input cell.
#pragma unroll
                for (uindex_stencil_t cache_c = 0; cache_c < uindex_stencil_t(stencil_diameter);
                     cache_c++) {
                    T new_value;
                    if (cache_c == uindex_stencil_t(stencil_diameter - 1)) {
                        new_value = carry;
                    } else {
                        new_value = cache[c[stage][0]][r[stage]][stage][cache_c].value;
                    }

                    stencil_buffer[stage][cache_c][stencil_diameter - 1] = new_value;
                    if (cache_c > 0) {
                        cache[(~c[stage])[0]][r[stage]][stage][cache_c - 1].value = new_value;
                    }
                }

                if (n_generations > i_generation + pipeline_length ||
                    stage < uindex_plen_t(n_generations - i_generation)) {
                    StencilImpl stencil(ID(c[stage], r[stage]), UID(grid_width, grid_height),
                                        i_generation + uindex_t(stage), uindex_t(stage));

                    bool h_halo_mask[stencil_diameter];
                    bool v_halo_mask[stencil_diameter];
#pragma unroll
                    for (uindex_stencil_t mask_i = 0; mask_i < uindex_stencil_t(stencil_diameter);
                         mask_i++) {
                        // These computation assume that the central cell is in the grid. If it's
                        // not, the resulting value of this stage will be discarded anyways, so this
                        // is safe.
                        if (mask_i < uindex_stencil_t(stencil_radius)) {
                            h_halo_mask[mask_i] = c[stage] >= index_1d_t(stencil_radius - mask_i);
                            v_halo_mask[mask_i] = r[stage] >= index_1d_t(stencil_radius - mask_i);
                        } else if (mask_i == uindex_stencil_t(stencil_radius)) {
                            h_halo_mask[mask_i] = true;
                            v_halo_mask[mask_i] = true;
                        } else {
                            h_halo_mask[mask_i] =
                                c[stage] < grid_width + index_1d_t(stencil_radius - mask_i);
                            v_halo_mask[mask_i] =
                                r[stage] < grid_height + index_1d_t(stencil_radius - mask_i);
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
                                    stencil_buffer[stage][cell_c][cell_r];
                            } else {
                                stencil[StencilUID(cell_c, cell_r)] = halo_value;
                            }
                        }
                    }

                    carry = trans_func(stencil);
                } else {
                    carry = stencil_buffer[stage][stencil_radius][stencil_radius];
                }

                r[stage] += 1;
                if (r[stage] == index_1d_t(grid_height)) {
                    r[stage] = 0;
                    c[stage] += 1;
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
    uindex_t n_generations;
    uindex_1d_t grid_width;
    uindex_1d_t grid_height;
    T halo_value;
};

} // namespace monotile
} // namespace stencil