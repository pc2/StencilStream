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

namespace stencil {
namespace tiling {

/**
 * \brief A kernel that executes a stencil transition function on a tile.
 *
 * It receives the contents of a tile and it's halo from the `in_pipe`, applies the transition
 * function when applicable and writes the result to the `out_pipe`.
 *
 * \tparam TransFunc The type of transition function to use.
 * \tparam n_processing_elements The number of processing elements to use. Similar to an unroll
 * factor for a loop.
 * \tparam output_tile_width The number of columns in a grid tile.
 * \tparam output_tile_height The number of rows in a grid tile.
 * \tparam in_pipe The pipe to read from.
 * \tparam out_pipe The pipe to write to.
 */
template <typename TransFunc, uindex_t n_processing_elements, uindex_t output_tile_width,
          uindex_t output_tile_height, typename in_pipe, typename out_pipe>
class ExecutionKernel {
  public:
    using Cell = typename TransFunc::Cell;

    static_assert(std::is_invocable_r<Cell, TransFunc const, Stencil<TransFunc> const &>::value);
    static_assert(TransFunc::stencil_radius >= 1);

    using StencilImpl = Stencil<TransFunc>;

    /**
     * \brief The width and height of the stencil buffer.
     */
    static constexpr uindex_t stencil_diameter = Stencil<TransFunc>::diameter;

    static constexpr uindex_t halo_radius = TransFunc::stencil_radius * n_processing_elements;

    /**
     * \brief The width of the processed tile with the tile halo attached.
     */
    static constexpr uindex_t max_input_tile_width = 2 * halo_radius + output_tile_width;

    /**
     * \brief The height of the processed tile with the tile halo attached.
     */
    static constexpr uindex_t input_tile_height = 2 * halo_radius + output_tile_height;

    /**
     * \brief The total number of cells to read from the `in_pipe`.
     */
    static constexpr uindex_t n_input_cells = max_input_tile_width * input_tile_height;

    using index_stencil_t = typename StencilImpl::index_stencil_t;
    using uindex_stencil_t = typename StencilImpl::uindex_stencil_t;
    using StencilID = typename StencilImpl::StencilID;
    using StencilUID = typename StencilImpl::StencilUID;

    static constexpr unsigned long bits_1d =
        std::bit_width(std::max(max_input_tile_width, input_tile_height));
    using index_1d_t = ac_int<bits_1d + 1, true>;
    using uindex_1d_t = ac_int<bits_1d, false>;

    static constexpr unsigned long bits_2d = 2 * bits_1d;
    using index_2d_t = ac_int<bits_2d + 1, true>;
    using uindex_2d_t = ac_int<bits_2d, false>;

    static constexpr unsigned long bits_pes = std::bit_width(n_processing_elements);
    using index_pes_t = ac_int<bits_pes + 1, true>;
    using uindex_pes_t = ac_int<bits_pes, false>;

    /**
     * \brief Create and configure the execution kernel.
     *
     * \param trans_func The instance of the transition function to use.
     * \param i_generation The generation index of the input cells.
     * \param target_i_generation The number of generations to compute. If this number is bigger
     * than `n_processing_elements`, only `n_processing_elements` generations will be computed.
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
                    uindex_t grid_height, Cell halo_value)
        : trans_func(trans_func), i_generation(i_generation),
          n_generations(
              std::min(uindex_t(n_processing_elements), target_i_generation - i_generation)),
          grid_c_offset(grid_c_offset), grid_r_offset(grid_r_offset), grid_width(grid_width),
          grid_height(grid_height), halo_value(halo_value) {
        assert(grid_c_offset % output_tile_width == 0);
        assert(grid_r_offset % output_tile_height == 0);
    }

    /**
     * \brief Execute the configured operations.
     */
    void operator()() const {
        uindex_1d_t input_tile_c = 0;
        uindex_1d_t input_tile_r = 0;

        /*
         * The intel::numbanks attribute requires a power of two as it's argument and if the
         * number of processing elements isn't a power of two, it would produce an error. Therefore,
         * we calculate the next power of two and use it to allocate the cache. The compiler is
         * smart enough to see that these additional banks in the cache aren't used and therefore
         * optimizes them away.
         */
        [[intel::fpga_memory,
          intel::numbanks(2 * std::bit_ceil(n_processing_elements))]] Padded<Cell>
            cache[2][input_tile_height][std::bit_ceil(n_processing_elements)][stencil_diameter - 1];
        [[intel::fpga_register]] Cell stencil_buffer[n_processing_elements][stencil_diameter]
                                                    [stencil_diameter];

        uindex_1d_t tile_section_width = std::min(output_tile_width, grid_width - grid_c_offset);
        uindex_1d_t tile_section_height = output_tile_height;
        uindex_2d_t n_iterations =
            (tile_section_width + 2 * halo_radius) * (tile_section_height + 2 * halo_radius);

        for (uindex_2d_t i = 0; i < n_iterations; i++) {
            [[intel::fpga_register]] Cell carry = in_pipe::read();

#pragma unroll
            for (uindex_pes_t i_processing_element = 0;
                 i_processing_element < uindex_pes_t(n_processing_elements);
                 i_processing_element++) {
                /*
                 * Shift up every value in the stencil_buffer.
                 * This operation does not touch the values in the bottom row, which will be filled
                 * from the cache and the new input value later.
                 */
#pragma unroll
                for (uindex_stencil_t r = 0; r < uindex_stencil_t(stencil_diameter - 1); r++) {
#pragma unroll
                    for (uindex_stencil_t c = 0; c < uindex_stencil_t(stencil_diameter); c++) {
                        stencil_buffer[i_processing_element][c][r] =
                            stencil_buffer[i_processing_element][c][r + 1];
                    }
                }

                index_1d_t rel_input_grid_c =
                    index_1d_t(input_tile_c) -
                    index_1d_t((stencil_diameter - 1) +
                               (n_processing_elements + i_processing_element - 2) *
                                   TransFunc::stencil_radius);
                index_t input_grid_c = grid_c_offset + rel_input_grid_c.to_int64();
                index_1d_t rel_input_grid_r =
                    index_1d_t(input_tile_r) -
                    index_1d_t((stencil_diameter - 1) +
                               (n_processing_elements + i_processing_element - 2) *
                                   TransFunc::stencil_radius);
                index_t input_grid_r = grid_r_offset + rel_input_grid_r.to_int64();

                // Update the stencil buffer and cache with previous cache contents and the new
                // input cell.
#pragma unroll
                for (uindex_stencil_t cache_c = 0; cache_c < uindex_stencil_t(stencil_diameter);
                     cache_c++) {
                    Cell new_value;
                    if (cache_c == uindex_stencil_t(stencil_diameter - 1)) {
                        bool is_halo = (grid_c_offset == 0 && rel_input_grid_c < 0);
                        is_halo |= (grid_r_offset == 0 && rel_input_grid_r < 0);
                        is_halo |= input_grid_c >= grid_width || input_grid_r >= grid_height;

                        new_value = is_halo ? halo_value : carry;
                    } else {
                        new_value =
                            cache[input_tile_c[0]][input_tile_r][i_processing_element][cache_c]
                                .value;
                    }

                    stencil_buffer[i_processing_element][cache_c][stencil_diameter - 1] = new_value;
                    if (cache_c > 0) {
                        cache[(~input_tile_c)[0]][input_tile_r][i_processing_element][cache_c - 1]
                            .value = new_value;
                    }
                }

                index_t output_grid_c = input_grid_c - index_t(TransFunc::stencil_radius);
                index_t output_grid_r = input_grid_r - index_t(TransFunc::stencil_radius);
                StencilImpl stencil(ID(output_grid_c, output_grid_r), UID(grid_width, grid_height),
                                    i_generation + i_processing_element.to_uint64(),
                                    i_processing_element.to_uint64(),
                                    stencil_buffer[i_processing_element]);

                if (i_processing_element.to_uint64() < n_generations) {
                    carry = trans_func(stencil);
                } else {
                    carry = stencil_buffer[i_processing_element][TransFunc::stencil_radius]
                                          [TransFunc::stencil_radius];
                }
            }

            bool is_valid_output =
                input_tile_c >= uindex_1d_t((stencil_diameter - 1) * n_processing_elements);
            is_valid_output &=
                input_tile_r >= uindex_1d_t((stencil_diameter - 1) * n_processing_elements);

            if (is_valid_output) {
                out_pipe::write(carry);
            }

            if (input_tile_r == uindex_1d_t(input_tile_height - 1)) {
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
    uindex_t n_generations;
    uindex_t grid_c_offset;
    uindex_t grid_r_offset;
    uindex_t grid_width;
    uindex_t grid_height;
    Cell halo_value;
};

} // namespace tiling
} // namespace stencil