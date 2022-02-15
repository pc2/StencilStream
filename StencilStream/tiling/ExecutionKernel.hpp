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
namespace tiling {

/**
 * \brief A kernel that executes a stencil transition function on a tile.
 *
 * It receives the contents of a tile and it's halo from the `in_pipe`, applies the transition
 * function when applicable and writes the result to the `out_pipe`.
 *
 * \tparam TransFunc The type of transition function to use.
 * \tparam T Cell value type.
 * \tparam stencil_radius The static, maximal Chebyshev distance of cells in a stencil to the
 * central cell
 * \tparam pipeline_length The number of pipeline stages to use. Similar to an unroll
 * factor for a loop.
 * \tparam output_tile_width The number of columns in a grid tile.
 * \tparam output_tile_height The number of rows in a grid tile.
 * \tparam in_pipe The pipe to read from.
 * \tparam out_pipe The pipe to write to.
 */
template <typename TransFunc, typename T, uindex_min_t stencil_radius, uindex_1d_t pipeline_length,
          uindex_1d_t output_tile_width, uindex_1d_t output_tile_height, typename in_pipe,
          typename out_pipe>
class ExecutionKernel {
  public:
    static_assert(
        std::is_invocable_r<T, TransFunc const, Stencil<T, stencil_radius> const &>::value);
    static_assert(stencil_radius >= 1);

    /**
     * \brief The width and height of the stencil buffer.
     */
    static constexpr uindex_min_t stencil_diameter = Stencil<T, stencil_radius>::diameter;

    /**
     * \brief The width of the processed tile with the tile halo attached.
     */
    static constexpr uindex_1d_t input_tile_width =
        BOUND_CHECK(2 * stencil_radius * pipeline_length + output_tile_width, uindex_1d_t);

    /**
     * \brief The height of the processed tile with the tile halo attached.
     */
    static constexpr uindex_1d_t input_tile_height =
        BOUND_CHECK(2 * stencil_radius * pipeline_length + output_tile_height, uindex_1d_t);

    /**
     * \brief The total number of cells to read from the `in_pipe`.
     */
    static constexpr uindex_2d_t n_input_cells =
        BOUND_CHECK(input_tile_width * input_tile_height, uindex_2d_t);

    /**
     * \brief Create and configure the execution kernel.
     *
     * \param trans_func The instance of the transition function to use.
     * \param i_generation The generation index of the input cells.
     * \param target_i_generation The number of generations to compute. If this number is bigger
     * than `pipeline_length`, only `pipeline_length` generations will be computed.
     * \param grid_c_offset The column offset of the processed tile relative to the grid's origin,
     * not including the halo. For example, for the most north-western tile the offset will always
     * be (0,0), not (-halo_radius,-halo_radius)
     * \param grid_r_offset The row offset of the processed tile relative to the grid's origin. See
     * `grid_c_offset` for details.
     * \param grid_width The number of cell columns in the grid.
     * \param grid_height The number of cell rows in the grid.
     * \param halo_value The value of cells in the grid halo.
     */
    ExecutionKernel(TransFunc trans_func, uindex_t i_generation, uindex_t target_i_generation,
                    uindex_t grid_c_offset, uindex_t grid_r_offset, uindex_t grid_width,
                    uindex_t grid_height, T halo_value)
        : trans_func(trans_func), i_generation(i_generation),
          n_generations(std::min(uindex_t(pipeline_length), target_i_generation - i_generation)),
          grid_c_offset(grid_c_offset), grid_r_offset(grid_r_offset), grid_width(grid_width),
          grid_height(grid_height), halo_value(halo_value) {}

    /**
     * \brief Execute the configured operations.
     */
    void operator()() const {
        uindex_1d_t input_tile_c = 0;
        uindex_1d_t input_tile_r = 0;

        /*
         * The intel::numbanks attribute requires a power of two as it's argument and if the
         * pipeline length isn't a power of two, it would produce an error. Therefore, we calculate
         * the next power of two and use it to allocate the cache. The compiler is smart enough to
         * see that these additional banks in the cache aren't used and therefore optimizes them
         * away.
         */
        [[intel::fpga_memory, intel::numbanks(2 * next_power_of_two(pipeline_length))]] T
            cache[2][input_tile_height][next_power_of_two(pipeline_length)][stencil_diameter - 1];
        [[intel::fpga_register]] T stencil_buffer[pipeline_length][stencil_diameter]
                                                 [stencil_diameter];

        for (uindex_2d_t i = 0; i < n_input_cells; i++) {
            T value = in_pipe::read();

#pragma unroll
            for (uindex_1d_t stage = 0; stage < pipeline_length; stage++) {
                /*
                 * Shift up every value in the stencil_buffer.
                 * This operation does not touch the values in the bottom row, which will be filled
                 * from the cache and the new input value later.
                 */
#pragma unroll
                for (uindex_min_t r = 0; r < stencil_diameter - 1; r++) {
#pragma unroll
                    for (uindex_min_t c = 0; c < stencil_diameter; c++) {
                        stencil_buffer[stage][c][r] = stencil_buffer[stage][c][r + 1];
                    }
                }

                index_t input_grid_c = grid_c_offset + index_t(input_tile_c) -
                                       (stencil_diameter - 1) -
                                       (pipeline_length + stage - 2) * stencil_radius;
                index_t input_grid_r = grid_r_offset + index_t(input_tile_r) -
                                       (stencil_diameter - 1) -
                                       (pipeline_length + stage - 2) * stencil_radius;

                // Update the stencil buffer and cache with previous cache contents and the new
                // input cell.
#pragma unroll
                for (uindex_min_t cache_c = 0; cache_c < stencil_diameter; cache_c++) {
                    T new_value;
                    if (cache_c == stencil_diameter - 1) {
                        if (input_grid_c < 0 || input_grid_r < 0 || input_grid_c >= grid_width ||
                            input_grid_r >= grid_height) {
                            new_value = halo_value;
                        } else {
                            new_value = value;
                        }
                    } else {
                        new_value = cache[input_tile_c & 0b1][input_tile_r][stage][cache_c];
                    }

                    stencil_buffer[stage][cache_c][stencil_diameter - 1] = new_value;
                    if (cache_c > 0) {
                        cache[(~input_tile_c) & 0b1][input_tile_r][stage][cache_c - 1] = new_value;
                    }
                }

                index_t output_grid_c = input_grid_c - stencil_radius;
                index_t output_grid_r = input_grid_r - stencil_radius;
                Stencil<T, stencil_radius> stencil(
                    ID(output_grid_c, output_grid_r), i_generation + stage, stage,
                    stencil_buffer[stage], UID(grid_width, grid_height));

                if (stage < n_generations) {
                    value = trans_func(stencil);
                } else {
                    value = stencil_buffer[stage][stencil_radius][stencil_radius];
                }
            }

            bool is_valid_output = input_tile_c >= (stencil_diameter - 1) * pipeline_length;
            is_valid_output &= input_tile_r >= (stencil_diameter - 1) * pipeline_length;

            if (is_valid_output) {
                out_pipe::write(value);
            }

            if (input_tile_r == input_tile_height - 1) {
                input_tile_r = 0;
                input_tile_c++;
            } else {
                input_tile_r++;
            }
        }
    }

  private:
    TransFunc trans_func;
    uindex_t i_generation;
    uindex_1d_t n_generations;
    uindex_t grid_c_offset;
    uindex_t grid_r_offset;
    uindex_t grid_width;
    uindex_t grid_height;
    T halo_value;
};

} // namespace tiling
} // namespace stencil