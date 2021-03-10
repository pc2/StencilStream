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

namespace stencil_stream
{

/**
 * A core that executes a stencil transition function on a grid.
 * 
 * This core iterates over the rows first and over the columns second due to the way the cells are fed to it by the IO kernels.
 * 
 * The columns that are covered by the stencil are stored in a cache and are used again when the iteration reaches a new coloum.
 */
template <typename T, UIndex stencil_radius, UIndex input_grid_width, UIndex input_grid_height>
class ExecutionCore
{
public:
    static_assert(stencil_radius >= 1);

    ExecutionCore() : input_column(0),
                      input_row(0),
                      output_column_offset(0),
                      output_row_offset(0),
                      cache(),
                      active_cache(0),
                      info()
    {
    }

    /**
     * Create a new execution core.
     * 
     * Due to technical problems with the dpcpp compiler, the cache can not be declared as a member of this
     * class. Instead, it has to be defined as a variable and be passed as a reference into the core object.
     */
    ExecutionCore(
        Index cell_generation,
        Index output_column_offset,
        Index output_row_offset) : input_column(0),
                                   input_row(0),
                                   output_column_offset(output_column_offset),
                                   output_row_offset(output_row_offset),
                                   cache(),
                                   active_cache(0),
                                   info()
    {
        info.cell_generation = cell_generation;
    }

    /**
     * Process the next input cell, execute the transition function and return the result.
     */
    template <typename TransFunc>
    T step(T input, TransFunc trans_func)
    {
        static_assert(
            std::is_invocable_r<T, TransFunc, Stencil<T, stencil_radius> const &, StencilInfo const &>::
                value);

        /**
         * Shift up every value in the stencil.
         * This operation does not touch the values in the bottom row, which will be filled
         * from the cache and the new input value later.
         */
#pragma unroll
        for (UIndex r = 0; r < stencil.diameter() - 1; r++)
        {
#pragma unroll
            for (UIndex c = 0; c < stencil.diameter(); c++)
            {
                stencil[UID(c, r)] = stencil[UID(c, r + 1)];
            }
        }

        // Update the stencil buffer and cache with previous cache contents and the new input cell.
#pragma unroll
        for (UIndex cache_c = 0; cache_c < stencil.diameter(); cache_c++)
        {
            T new_value;
            if (cache_c == stencil.diameter() - 1)
            {
                new_value = input;
            }
            else
            {
                new_value = cache[active_cache][input_row][cache_c];
            }

            stencil[UID(cache_c, stencil.diameter() - 1)] = new_value;
            if (cache_c > 0)
            {
                cache[passive_cache()][input_row][cache_c - 1] = new_value;
            }
        }

        Index output_column = get_output_column();
        Index output_row = get_output_row();
        info.center_cell_id = ID(output_column, output_row);

        T output = trans_func(stencil, info);

        // Increase column and row counters.
        if (input_row == input_grid_height - 1)
        {
            active_cache = passive_cache();
            input_row = 0;
            if (input_column == input_grid_width - 1)
            {
                input_column = 0;
            }
            else
            {
                input_column++;
            }
        }
        else
        {
            input_row++;
        }

        return output;
    }

    UIndex get_input_column() const
    {
        return input_column;
    }

    UIndex get_input_row() const
    {
        return input_row;
    }

    Index get_output_column() const
    {
        return input_column - stencil_radius + output_column_offset;
    }

    Index get_output_row() const
    {
        return input_row - stencil_radius + output_row_offset;
    }

private:
    UIndex passive_cache() const
    {
        return active_cache == 0 ? 1 : 0;
    }

    [[intel::fpga_register]] UIndex input_column;
    [[intel::fpga_register]] UIndex input_row;
    [[intel::fpga_register]] Index output_column_offset;
    [[intel::fpga_register]] Index output_row_offset;

    [[intel::fpga_memory, intel::numbanks(2)]] T cache[2][input_grid_height][Stencil<T, stencil_radius>::diameter() - 1];
    [[intel::fpga_register]] UIndex active_cache;

    [[intel::fpga_register]] Stencil<T, stencil_radius> stencil;
    [[intel::fpga_register]] StencilInfo info;
};

} // namespace stencil_stream