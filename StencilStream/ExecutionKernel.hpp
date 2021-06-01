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
#include "StencilInfo.hpp"
#include <optional>

namespace stencil
{

/**
 * A core that executes a stencil transition function on a tile.
 * 
 * This core iterates over the rows first and over the columns second due to the way the cells are fed to it by the IO kernels.
 * 
 * The columns that are covered by the stencil are stored in a cache and are used again when the iteration reaches a new coloum.
 */
template <typename TransFunc, typename T, uindex_t stencil_radius, uindex_t pipeline_length, uindex_t output_tile_width, uindex_t output_tile_height, typename in_pipe, typename out_pipe>
class ExecutionKernel
{
public:
    static_assert(
        std::is_invocable_r<T, TransFunc const, Stencil<T, stencil_radius> const &, StencilInfo const &>::
            value);
    static_assert(stencil_radius >= 1);

    const static uindex_t stencil_diameter = Stencil<T, stencil_radius>::diameter();
    const static uindex_t n_input_cells = (2 * stencil_radius * pipeline_length + output_tile_width) * (2 * stencil_radius * pipeline_length + output_tile_height);
    const static uindex_t input_tile_width = 2 * stencil_radius * pipeline_length + output_tile_width;
    const static uindex_t input_tile_height = 2 * stencil_radius * pipeline_length + output_tile_height;

    ExecutionKernel(
        TransFunc trans_func,
        uindex_t i_generation,
        uindex_t n_generations,
        uindex_t grid_c_offset,
        uindex_t grid_r_offset,
        uindex_t grid_width,
        uindex_t grid_height,
        T halo_value) : trans_func(trans_func),
                        i_generation(i_generation),
                        n_generations(n_generations),
                        grid_c_offset(grid_c_offset),
                        grid_r_offset(grid_r_offset),
                        grid_width(grid_width),
                        grid_height(grid_height),
                        halo_value(halo_value)
    {
    }

    void operator()() const
    {
        uindex_t input_tile_c = 0;
        uindex_t input_tile_r = 0;

        [[intel::fpga_memory, intel::numbanks(2 * pipeline_length)]] T cache[2][input_tile_height][pipeline_length][stencil_diameter - 1];
        uindex_t active_cache = 0;
        [[intel::fpga_register]] T stencil[pipeline_length][stencil_diameter][stencil_diameter];

        for (uindex_t i = 0; i < n_input_cells; i++)
        {
            T value = in_pipe::read();

#pragma unroll
            for (uindex_t stage = 0; stage < pipeline_length; stage++)
            {
                /**
                 * Shift up every value in the stencil.
                 * This operation does not touch the values in the bottom row, which will be filled
                 * from the cache and the new input value later.
                 */
#pragma unroll
                for (uindex_t r = 0; r < stencil_diameter - 1; r++)
                {
#pragma unroll
                    for (uindex_t c = 0; c < stencil_diameter; c++)
                    {
                        stencil[stage][c][r] = stencil[stage][c][r + 1];
                    }
                }

                index_t input_grid_c = grid_c_offset + index_t(input_tile_c) - (stencil_diameter - 1) - (pipeline_length + stage - 2) * stencil_radius;
                index_t input_grid_r = grid_r_offset + index_t(input_tile_r) - (stencil_diameter - 1) - (pipeline_length + stage - 2) * stencil_radius;

                // Update the stencil buffer and cache with previous cache contents and the new input cell.
#pragma unroll
                for (uindex_t cache_c = 0; cache_c < stencil_diameter; cache_c++)
                {
                    T new_value;
                    if (cache_c == stencil_diameter - 1)
                    {
                        if (input_grid_c < 0 || input_grid_r < 0 || input_grid_c >= grid_width || input_grid_r >= grid_height)
                        {
                            new_value = halo_value;
                        }
                        else
                        {
                            new_value = value;
                        }
                    }
                    else
                    {
                        new_value = cache[active_cache][input_tile_r][stage][cache_c];
                    }

                    stencil[stage][cache_c][stencil_diameter - 1] = new_value;
                    if (cache_c > 0)
                    {
                        cache[active_cache == 0 ? 1 : 0][input_tile_r][stage][cache_c - 1] = new_value;
                    }
                }

                index_t output_grid_c = input_grid_c - stencil_radius;
                index_t output_grid_r = input_grid_r - stencil_radius;
                StencilInfo info{
                    ID(output_grid_c, output_grid_r),
                    i_generation + stage,
                };

                if (stage < n_generations)
                {
                    value = trans_func(Stencil<T, stencil_radius>(stencil[stage]), info);
                }
                else
                {
                    value = stencil[stage][stencil_radius][stencil_radius];
                }
            }

            bool is_valid_output = input_tile_c >= (stencil_diameter - 1) * pipeline_length;
            is_valid_output &= input_tile_r >= (stencil_diameter - 1) * pipeline_length;

            if (is_valid_output)
            {
                out_pipe::write(value);
            }

            if (input_tile_r == input_tile_height - 1)
            {
                active_cache = active_cache == 0 ? 1 : 0;
                input_tile_r = 0;
                if (input_tile_c == input_tile_width - 1)
                {
                    input_tile_c = 0;
                }
                else
                {
                    input_tile_c++;
                }
            }
            else
            {
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
    T halo_value;
};

} // namespace stencil