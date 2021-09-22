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
 * \tparam T Cell value type.
 * \tparam stencil_radius The static, maximal Chebyshev distance of cells in a stencil to the
 * central cell \tparam pipeline_length The number of pipeline stages to use. Similar to an unroll
 * factor for a loop. \tparam output_tile_width The number of columns in a grid tile. \tparam
 * output_tile_height The number of rows in a grid tile. \tparam in_pipe The pipe to read from.
 * \tparam out_pipe The pipe to write to.
 * \tparam access_target The target/location were the input and output buffers are stored. The
 * default is `global_buffer`, which is the normal global memory of a SYCL device. If the kernel
 * should is used by the host, the access target should be `host_buffer`.
 */
template <typename TransFunc, typename T, uindex_t stencil_radius, uindex_t pipeline_length,
          uindex_t tile_width, uindex_t tile_height,
          cl::sycl::access::target access_target = cl::sycl::access::target::global_buffer>
class ExecutionKernel {
  public:
    static_assert(
        std::is_invocable_r<T, TransFunc const, Stencil<T, stencil_radius> const &>::value);
    static_assert(stencil_radius >= 1);

    /**
     * \brief Type of the input buffer accessor.
     */
    using InAccessor = cl::sycl::accessor<T, 2, cl::sycl::access::mode::read, access_target>;

    /**
     * \brief Type of the output buffer accessor.
     */
    using OutAccessor =
        cl::sycl::accessor<T, 2, cl::sycl::access::mode::discard_write, access_target>;

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
     * \param in_accessor The grid buffer of the input. The grid width/height is derived from this
     * accessor; It may not exceed the tile width/height. \param out_accessor The grid buffer of the
     * output. It has to have the same range as the `in_accessor`. \param trans_func The instance of
     * the transition function to use. \param i_generation The generation index of the input cells.
     * \param n_generations The number of generations to compute. If this number is bigger than
     * `pipeline_length`, only `pipeline_length` generations will be computed.
     * \param halo_value The value of cells outside the grid.
     */
    ExecutionKernel(InAccessor in_accessor, OutAccessor out_accessor, TransFunc trans_func,
                    uindex_t i_generation, uindex_t n_generations, T halo_value)
        : in_accessor(in_accessor), out_accessor(out_accessor), trans_func(trans_func),
          i_generation(i_generation), n_generations(n_generations),
          grid_width(in_accessor.get_range()[0]), grid_height(in_accessor.get_range()[1]),
          halo_value(halo_value) {
#ifndef __SYCL_DEVICE_ONLY__
        assert(grid_width <= tile_width);
        assert(grid_height <= tile_height);
        assert(in_accessor.get_range()[0] == in_accessor.get_range()[0]);
        assert(out_accessor.get_range()[1] == out_accessor.get_range()[1]);
#endif
    }

    /**
     * \brief Execute the kernel.
     */
    [[intel::kernel_args_restrict]] void operator()() const {
        [[intel::fpga_register]] index_t c[pipeline_length + 1];
        [[intel::fpga_register]] index_t r[pipeline_length + 1];

        // Initializing (output) column and row counters.
        c[0] = 0;
        r[0] = 0;
#pragma unroll
        for (uindex_t i = 1; i < pipeline_length + 1; i++) {
            c[i] = c[i - 1] - stencil_radius;
            r[i] = r[i - 1] - stencil_radius;
            if (r[i] < index_t(0)) {
                r[i] += tile_height;
                c[i] -= 1;
            }
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

        for (uindex_t i_iteration = 0; i_iteration < n_iterations; i_iteration++) {
            T value;
            if (c[0] < index_t(grid_width) && r[0] < index_t(grid_height)) {
                value = in_accessor[c[0]][r[0]];
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
                        new_value = cache[c[stage + 1] & 0b1][r[stage + 1]][stage][cache_c];
                    }

                    stencil_buffer[stage][cache_c][stencil_diameter - 1] = new_value;
                    if (cache_c > 0) {
                        cache[(~c[stage + 1]) & 0b1][r[stage + 1]][stage][cache_c - 1] = new_value;
                    }
                }

                if (i_generation + stage < n_generations) {
                    if (id_in_grid(c[stage + 1], r[stage + 1])) {
                        Stencil<T, stencil_radius> stencil(ID(c[stage + 1], r[stage + 1]),
                                                           i_generation + stage, stage,
                                                           UID(grid_width, grid_height));

#pragma unroll
                        for (index_t cell_c = -stencil_radius; cell_c <= index_t(stencil_radius);
                             cell_c++) {
#pragma unroll
                            for (index_t cell_r = -stencil_radius;
                                 cell_r <= index_t(stencil_radius); cell_r++) {
                                if (id_in_grid(cell_c + c[stage + 1], cell_r + r[stage + 1])) {
                                    stencil[ID(cell_c, cell_r)] =
                                        stencil_buffer[stage][cell_c + stencil_radius]
                                                      [cell_r + stencil_radius];
                                } else {
                                    stencil[ID(cell_c, cell_r)] = halo_value;
                                }
                            }
                        }

                        value = trans_func(stencil);
                    } else {
                        value = halo_value;
                    }
                } else {
                    value = stencil_buffer[stage][stencil_radius][stencil_radius];
                }
            }

            if (c[pipeline_length] >= index_t(0) && c[pipeline_length] < index_t(grid_width) &&
                r[pipeline_length] < index_t(grid_height)) {
                out_accessor[c[pipeline_length]][r[pipeline_length]] = value;
            }

#pragma unroll
            for (uindex_t i = 0; i < pipeline_length + 1; i++) {
                r[i] += 1;
                if (r[i] == tile_height) {
                    r[i] = 0;
                    c[i] += 1;
                }
            }
        }
    }

  private:
    bool id_in_grid(index_t c, index_t r) const {
        return c >= index_t(0) && r >= index_t(0) && c < index_t(grid_width) &&
               r < index_t(grid_height);
    }

    InAccessor in_accessor;
    OutAccessor out_accessor;
    TransFunc trans_func;
    uindex_t i_generation;
    uindex_t n_generations;
    uindex_t grid_width;
    uindex_t grid_height;
    T halo_value;
};

} // namespace monotile
} // namespace stencil