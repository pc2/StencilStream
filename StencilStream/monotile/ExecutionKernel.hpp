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
#pragma once
#include "../GenericID.hpp"
#include "../Helpers.hpp"
#include "../Index.hpp"
#include "../Stencil.hpp"
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

    /**
     * \brief The width and height of the stencil buffer.
     */
    const static uindex_t stencil_diameter = Stencil<T, stencil_radius>::diameter;

    /**
     * \brief The number of cells in the tile.
     */
    const static uindex_t n_cells = tile_width * tile_height;

    /**
     * \brief The number of cells that need to be fed into a stage before it produces correct
     * values.
     */
    const static uindex_t stage_latency = stencil_radius * (tile_height + 1);

    /**
     * \brief The number of cells that need to be fed into the pipeline before it produces correct
     * values.
     */
    const static uindex_t pipeline_latency = pipeline_length * stage_latency;

    /**
     * \brief The total number of loop iterations.
     */
    const static uindex_t n_iterations = pipeline_latency + n_cells;

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
                    uindex_t grid_width, uindex_t grid_height, T halo_value)
        : trans_func(trans_func), i_generation(i_generation), n_generations(n_generations),
          grid_width(grid_width), grid_height(grid_height), halo_value(halo_value) {}

    /**
     * \brief Execute the kernel.
     */
    void operator()() const {
        [[intel::fpga_register]] index_t c[pipeline_length];
        [[intel::fpga_register]] index_t r[pipeline_length];

        // Initializing (output) column and row counters.
        index_t prev_c = 0;
        index_t prev_r = 0;
#pragma unroll
        for (uindex_t i = 0; i < pipeline_length; i++) {
            c[i] = prev_c - stencil_radius;
            r[i] = prev_r - stencil_radius;
            if (r[i] < index_t(0)) {
                r[i] += tile_height;
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
        [[intel::fpga_memory, intel::numbanks(2 * next_power_of_two(pipeline_length))]] T
            cache[2][tile_height][next_power_of_two(pipeline_length)][stencil_diameter - 1];
        [[intel::fpga_register]] T stencil_buffer[pipeline_length][stencil_diameter]
                                                 [stencil_diameter];

        for (uindex_t i = 0; i < n_iterations; i++) {
            T value;
            if (i < n_cells) {
                value = in_pipe::read();
            } else {
                value = halo_value;
            }

#pragma unroll
            for (uindex_t stage = 0; stage < pipeline_length; stage++) {
#pragma unroll
                for (uindex_t r = 0; r < stencil_diameter - 1; r++) {
#pragma unroll
                    for (uindex_t c = 0; c < stencil_diameter; c++) {
                        stencil_buffer[stage][c][r] = stencil_buffer[stage][c][r + 1];
                    }
                }

                // Update the stencil buffer and cache with previous cache contents and the new
                // input cell.
#pragma unroll
                for (uindex_t cache_c = 0; cache_c < stencil_diameter; cache_c++) {
                    T new_value;
                    if (cache_c == stencil_diameter - 1) {
                        new_value = value;
                    } else {
                        new_value = cache[c[stage] & 0b1][r[stage]][stage][cache_c];
                    }

                    stencil_buffer[stage][cache_c][stencil_diameter - 1] = new_value;
                    if (cache_c > 0) {
                        cache[(~c[stage]) & 0b1][r[stage]][stage][cache_c - 1] = new_value;
                    }
                }

                if (i_generation + stage < n_generations) {
                    Stencil<T, stencil_radius> stencil(ID(c[stage], r[stage]),
                                                       i_generation + stage, stage,
                                                       UID(grid_width, grid_height));

                    bool h_halo_mask[stencil_diameter];
                    bool v_halo_mask[stencil_diameter];
                    #pragma unroll
                    for (index_t mask_i = 0; mask_i < stencil_diameter; mask_i++) {
                        // These computation assume that the central cell is in the grid. If it's not,
                        // the resulting value of this stage will be discarded anyways, so this is safe.
                        if (mask_i < stencil_radius) {
                            h_halo_mask[mask_i] = c[stage] >= index_t(stencil_radius) - mask_i;
                            v_halo_mask[mask_i] = r[stage] >= index_t(stencil_radius) - mask_i;
                        } else if (mask_i == stencil_radius) {
                            h_halo_mask[mask_i] = true;
                            v_halo_mask[mask_i] = true;
                        } else {
                            h_halo_mask[mask_i] = c[stage] < index_t(grid_width) + index_t(stencil_radius) - mask_i;
                            v_halo_mask[mask_i] = r[stage] < index_t(grid_height) + index_t(stencil_radius) - mask_i;
                        }
                    }

#pragma unroll
                    for (uindex_t cell_c = 0; cell_c < stencil_diameter; cell_c++) {
#pragma unroll
                        for (uindex_t cell_r = 0; cell_r < stencil_diameter; cell_r++) {
                            if (h_halo_mask[cell_c] && v_halo_mask[cell_r]) {
                                stencil[UID(cell_c, cell_r)] = stencil_buffer[stage][cell_c][cell_r];
                            } else {
                                stencil[UID(cell_c, cell_r)] = halo_value;
                            }
                        }
                    }

                    value = trans_func(stencil);
                } else {
                    value = stencil_buffer[stage][stencil_radius][stencil_radius];
                }

                r[stage] += 1;
                if (r[stage] == tile_height) {
                    r[stage] = 0;
                    c[stage] += 1;
                }
            }

            if (i >= pipeline_latency) {
                out_pipe::write(value);
            }
        }
    }

  private:
    bool id_in_grid(index_t c, index_t r) const {
        return c >= index_t(0) && r >= index_t(0) && c < index_t(grid_width) &&
               r < index_t(grid_height);
    }

    TransFunc trans_func;
    uindex_t i_generation;
    uindex_t n_generations;
    uindex_t grid_width;
    uindex_t grid_height;
    T halo_value;
};

} // namespace monotile
} // namespace stencil