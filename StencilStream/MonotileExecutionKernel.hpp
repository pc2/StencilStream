/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include "GenericID.hpp"
#include "Index.hpp"
#include "Stencil.hpp"
#include <optional>

namespace stencil
{

/**
 * \brief A kernel that executes a stencil transition function on a tile.
 * 
 * It receives the contents of a tile and it's halo from the `in_pipe`, applies the transition function when applicable and writes the result to the `out_pipe`.
 * 
 * \tparam TransFunc The type of transition function to use.
 * \tparam T Cell value type.
 * \tparam stencil_radius The static, maximal Chebyshev distance of cells in a stencil to the central cell
 * \tparam pipeline_length The number of pipeline stages to use. Similar to an unroll factor for a loop.
 * \tparam output_tile_width The number of columns in a grid tile.
 * \tparam output_tile_height The number of rows in a grid tile.
 * \tparam in_pipe The pipe to read from.
 * \tparam out_pipe The pipe to write to.
 */
template <typename TransFunc, typename T, uindex_t stencil_radius, uindex_t pipeline_length, uindex_t tile_width, uindex_t tile_height, typename in_pipe, typename out_pipe>
class MonotileExecutionKernel
{
public:
    static_assert(
        std::is_invocable_r<T, TransFunc const, Stencil<T, stencil_radius> const &>::
            value);
    static_assert(stencil_radius >= 1);

    /**
     * \brief The width and height of the stencil buffer.
     */
    const static uindex_t stencil_diameter = Stencil<T, stencil_radius>::diameter;

    const static uindex_t n_cells = tile_width * tile_height;

    const static uindex_t stage_latency = stencil_radius * (tile_height + 1);

    const static uindex_t pipeline_latency = pipeline_length * stage_latency;

    /**
     * \brief The total number of cells to read from the `in_pipe`.
     */
    const static uindex_t n_iterations = pipeline_latency + n_cells;

    /**
     * \brief Create and configure the execution kernel.
     * 
     * \param trans_func The instance of the transition function to use.
     * \param i_generation The generation index of the input cells.
     * \param n_generations The number of generations to compute. If this number is bigger than `pipeline_length`, only `pipeline_length` generations will be computed.
     * \param grid_width The number of cell columns in the grid.
     * \param grid_height The number of cell rows in the grid.
     */
    MonotileExecutionKernel(
        TransFunc trans_func,
        uindex_t i_generation,
        uindex_t n_generations,
        uindex_t grid_width,
        uindex_t grid_height,
        T halo_value) : trans_func(trans_func),
                        i_generation(i_generation),
                        n_generations(n_generations),
                        grid_width(grid_width),
                        grid_height(grid_height),
                        halo_value(halo_value)
    {
    }

    void operator()() const
    {
        index_t c[pipeline_length];
        index_t r[pipeline_length];

        // Initializing (output) column and row counters.
        index_t prev_c = 0;
        index_t prev_r = 0;
#pragma unroll
        for (uindex_t i = 0; i < pipeline_length; i++)
        {
            c[i] = prev_c - stencil_radius;
            r[i] = prev_r - stencil_radius;
            if (r[i] < index_t(0))
            {
                r[i] += tile_height;
                c[i] -= 1;
            }
            prev_c = c[i];
            prev_r = r[i];
        }

        [[intel::fpga_memory, intel::numbanks(2 * pipeline_length)]] T cache[2][tile_height][pipeline_length][stencil_diameter - 1];
        [[intel::fpga_register]] T stencil_buffer[pipeline_length][stencil_diameter][stencil_diameter];

        // Initializing the first cache. All following caches will be initialized with their predecessor's output,
        // but since the first stage has no predecessor, it's cache needs to be initilized.
#pragma unroll
        for (uindex_t cache_r = 0; cache_r < tile_height; cache_r++)
        {
#pragma unroll
            for (uindex_t cache_c = 0; cache_c < stencil_diameter - 1; cache_c++)
            {
                cache[0][cache_r][0][cache_c] = cache[1][cache_r][0][cache_c] = halo_value;
            }
        }

        for (uindex_t i = 0; i < n_iterations; i++)
        {
            T value;
            if (i < n_cells)
            {
                value = in_pipe::read();
            }
            else
            {
                value = halo_value;
            }

#pragma unroll
            for (uindex_t stage = 0; stage < pipeline_length; stage++)
            {
#pragma unroll
                for (uindex_t r = 0; r < stencil_diameter - 1; r++)
                {
#pragma unroll
                    for (uindex_t c = 0; c < stencil_diameter; c++)
                    {
                        stencil_buffer[stage][c][r] = stencil_buffer[stage][c][r + 1];
                    }
                }

                // Update the stencil buffer and cache with previous cache contents and the new input cell.
#pragma unroll
                for (uindex_t cache_c = 0; cache_c < stencil_diameter; cache_c++)
                {
                    T new_value;
                    if (cache_c == stencil_diameter - 1)
                    {
                        new_value = value;
                    }
                    else
                    {
                        new_value = cache[c[stage] & 0b1][r[stage]][stage][cache_c];
                    }

                    stencil_buffer[stage][cache_c][stencil_diameter - 1] = new_value;
                    if (cache_c > 0)
                    {
                        cache[(~c[stage]) & 0b1][r[stage]][stage][cache_c - 1] = new_value;
                    }
                }

                if (i_generation + stage < n_generations)
                {
                    if (id_in_grid(c[stage], r[stage]))
                    {
                        Stencil<T, stencil_radius> stencil(
                            ID(c[stage], r[stage]),
                            i_generation + stage,
                            stage,
                            UID(grid_width, grid_height));

                        #pragma unroll
                        for (index_t cell_c = -stencil_radius; cell_c <= index_t(stencil_radius); cell_c++)
                        {
                            #pragma unroll
                            for (index_t cell_r = -stencil_radius; cell_r <= index_t(stencil_radius); cell_r++)
                            {
                                if (id_in_grid(cell_c + c[stage], cell_r + r[stage]))
                                {
                                    stencil[ID(cell_c, cell_r)] = stencil_buffer[stage][cell_c + stencil_radius][cell_r + stencil_radius];
                                }
                                else
                                {
                                    stencil[ID(cell_c, cell_r)] = halo_value;
                                }
                            }
                        }

                        value = trans_func(stencil);
                    }
                    else
                    {
                        value = halo_value;
                    }
                }
                else
                {
                    value = stencil_buffer[stage][stencil_radius][stencil_radius];
                }

                r[stage] += 1;
                if (r[stage] == tile_height)
                {
                    r[stage] = 0;
                    c[stage] += 1;
                }
            }

            if (i >= pipeline_latency)
            {
                out_pipe::write(value);
            }
        }
    }

private:
    bool id_in_grid(index_t c, index_t r) const
    {
        return c >= index_t(0) && r >= index_t(0) && c < index_t(grid_width) && r < index_t(grid_height);
    }

    TransFunc trans_func;
    uindex_t i_generation;
    uindex_t n_generations;
    uindex_t grid_width;
    uindex_t grid_height;
    T halo_value;
};

} // namespace stencil