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
 * A core that executes a stencil kernel on a grid.
 * 
 * This core iterates over the rows first and over the columns second due to the way the cells are fed to it by the IO kernels.
 * 
 * The columns that are covered by the stencil are stored in a cache and are used again when the iteration reaches a new coloum.
 */
template <typename T, UIndex kernel_radius, UIndex max_input_grid_height>
class ExecutionCore
{
public:
    static_assert(kernel_radius >= 1);

    ExecutionCore() : input_column(0),
                      input_row(0),
                      output_grid_width(0),
                      output_grid_height(0),
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
        UIndex cell_generation,
        UIndex output_grid_width,
        UIndex output_grid_height,
        Index output_column_offset,
        Index output_row_offset) : input_column(0),
                                   input_row(0),
                                   output_grid_width(output_grid_width),
                                   output_grid_height(output_grid_height),
                                   output_column_offset(output_column_offset),
                                   output_row_offset(output_row_offset),
                                   cache(),
                                   active_cache(0),
                                   info()
    {
        assert(max_input_grid_height >= 2 * kernel_radius + output_grid_height);
        info.cell_generation = cell_generation;
    }

    /**
     * Process the next input cell, execute the stencil kernel and return the result.
     */
    template <typename Kernel>
    std::optional<T> step(std::optional<T> input, Kernel kernel)
    {
        static_assert(
            std::is_invocable_r<T, Kernel, Stencil<T, kernel_radius> const &, StencilInfo const &>::
                value);
        if (!input.has_value())
        {
            return input;
        }

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
                new_value = *input;
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

        std::optional<T> output;

        if (input_column >= stencil.diameter() - 1 && input_row >= stencil.diameter() - 1)
        {
            Index output_column = input_column - (stencil.diameter() - 1) + output_column_offset;
            Index output_row = input_row - (stencil.diameter() - 1) + output_row_offset;
            info.center_cell_id = UID(output_column, output_row);

            output = kernel(stencil, info);
        }
        else
        {
            output = std::nullopt;
        }

        // Increase column and row counters.
        if (input_row == input_grid_height() - 1)
        {
            active_cache = passive_cache();
            input_row = 0;
            if (input_column == input_grid_width() - 1)
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

private:
    UIndex passive_cache() const
    {
        return active_cache == 0 ? 1 : 0;
    }

    UIndex input_grid_height() const
    {
        return 2 * kernel_radius + output_grid_height;
    }

    UIndex input_grid_width() const
    {
        return 2 * kernel_radius + output_grid_width;
    }

    [[intel::fpga_register]] UIndex input_column;
    [[intel::fpga_register]] UIndex input_row;
    [[intel::fpga_register]] UIndex output_grid_width;
    [[intel::fpga_register]] UIndex output_grid_height;
    [[intel::fpga_register]] Index output_column_offset;
    [[intel::fpga_register]] Index output_row_offset;

    [[intel::fpga_memory, intel::numbanks(2)]] T cache[2][max_input_grid_height][Stencil<T, kernel_radius>::diameter() - 1];
    [[intel::fpga_register]] UIndex active_cache;

    [[intel::fpga_register]] Stencil<T, kernel_radius> stencil;
    [[intel::fpga_register]] StencilInfo info;
};

} // namespace stencil