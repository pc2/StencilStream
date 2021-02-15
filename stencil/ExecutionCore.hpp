/*
 * Copyright © 2020 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
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
#include "pipeline_length.hpp"

namespace stencil
{

/**
 * A core that executes a stencil kernel on a grid.
 * 
 * This core iterates over the rows first and over the columns second due to the way the cells are fed to it by the IO kernels.
 * 
 * The columns that are covered by the stencil are stored in a cache and are used again when the iteration reaches a new coloum.
 */
template <typename T, UIndex radius, UIndex grid_width, UIndex grid_height, UIndex position, typename Kernel>
class ExecutionCore
{
public:
    static_assert(
        std::is_invocable_r<T, Kernel, Stencil<T, radius> const &, StencilInfo const &>::
            value);
    static_assert(radius >= 1);

    /**
     * The number of steps the core needs to process one generation of the grid.
     */
    static constexpr UIndex steps_per_matrix = grid_width * grid_height;

    /**
     * The latency of an execution core.
     * 
     * Since `ExecutionCore`s need at least the central cell in their stencil buffer to produce useful outputs, they need to
     * be stepped `LATENCY` times with valid cells to "warm-up". In this context, this number of steps is called "latency".
     */
    static constexpr UIndex latency = (grid_height + 1) * radius;

    /**
     * The number of pipeline loop bodies that have to be executed before all previous cores in the
     * pipeline produce useful values.
     * 
     * Since this core is the `position`th core in the pipeline, all previous cores produce useless
     * outputs until each of their latency steps have passed. Therefore, the whole execution
     * loop needs to be executed `WAITING_STEPS` times until this core receives useful values.
     */
    static constexpr UIndex warmup_steps = position * latency;

    /**
     * The initial, linear counting offset for the position of the input cell.
     * 
     * To simplify their code, their is no "warm-up state" for cores; The stepping code is always fully executed.
     * Cores keep track of the input cell's position by simply counting indices, but since the the first (position - 1) * LATENCY
     * input cells aren't valid cell, these have to be left out. Therefore, cores start counting from `STARTING_POINT` and hit
     * index 0 when the first valid input cell arrives.
     */
    static constexpr UIndex starting_point = warmup_steps == 0 ? 0 : (grid_width * grid_height) - warmup_steps;

    /**
     * Create a new execution core.
     * 
     * Due to technical problems with the dpcpp compiler, the cache can not be declared as a member of this
     * class. Instead, it has to be defined as a variable and be passed as a reference into the core object.
     */
    ExecutionCore(T (&cache)[2][grid_height][Stencil<T, radius>::diameter() - 1], UIndex n_generations, Kernel kernel)
        : n_generations(n_generations), input_id(starting_point / grid_height, starting_point % grid_height), cache(cache), active_cache(0), kernel(kernel), info()
    {
        info.cell_generation = Index(position) - Index(pipeline_length);
    }

    /**
     * Process the next input cell, execute the stencil kernel and return the result.
     */
    T step(T input)
    {
        UIndex &column = input_id.c;
        UIndex &row = input_id.r;

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
                new_value = cache[active_cache][row][cache_c];
            }

            stencil[UID(cache_c, stencil.diameter())] = new_value;
            if (cache_c > 0)
            {
                cache[passive_cache()][row][cache_c - 1] = new_value;
            }
        }

        // Calculate the column and row of the central cell in the stencil buffer.
        // This also handles the edge cases when the index transitions to a new row or to a new
        // generation.
        Index kernel_c = column - radius;
        Index kernel_r = row - radius;
        if (kernel_r < 0)
        {
            kernel_r += grid_height;
            kernel_c -= 1;
        }
        if (kernel_c < 0)
        {
            kernel_c += grid_width;
        }

        // Prepare the info for the stencil kernel.
        info.center_cell_id = UID(kernel_c, kernel_r);
        if (kernel_c == 0 && kernel_r == 0)
        {
            info.cell_generation += pipeline_length;
        }

        // Run the kernel and write the result as output.
        T output = kernel(stencil, info);

        // Increase column and row counters.
        row++;
        if (row == grid_height)
        {
            active_cache = passive_cache();
            row = 0;
            column++;
            if (column == grid_width)
            {
                column = 0;
            }
        }

        // Return the output if it's wanted. If the total number of generations to compute isn't a
        // multiple of the pipeline length, some of the last generations must be discarded.
        if (info.cell_generation < n_generations)
        {
            return output;
        }
        else
        {
            return stencil[ID(0, 0)];
        }
    }

private:
    UIndex passive_cache() const
    {
        return active_cache == 0 ? 1 : 0;
    }

    UIndex n_generations;
    UID input_id;

    T(&cache)
    [2][grid_height][Stencil<T, radius>::diameter() - 1];
    UIndex active_cache;

    Kernel kernel;
    Stencil<T, radius> stencil;
    StencilInfo info;
};

} // namespace stencil