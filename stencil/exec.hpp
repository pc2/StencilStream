#pragma once
#include "data.hpp"
#include "io.hpp"
#include "pregen.hpp"

namespace stencil
{

/**
 * A core that executes a stencil kernel on a grid.
 * 
 * This core iterates over the rows first and over the columns second due to the way the cells are fed to it by the IO kernels.
 * 
 * The columns that are covered by the stencil are stored in a cache and are used again when the iteration reaches a new coloum.
 */
template <typename T, uindex_t radius, uindex_t grid_width, uindex_t grid_height, uindex_t position, typename Kernel>
class ExecutionCore
{
public:
    static_assert(
        std::is_invocable_r<T, Kernel, Stencil2D<T, radius> const &, Stencil2DInfo const &>::
            value);
    static_assert(radius >= 1);

    /**
     * The number of steps the core needs to process one generation of the grid.
     */
    static constexpr uindex_t steps_per_matrix = grid_width * grid_height;

    /**
     * The latency of an execution core.
     * 
     * Since `ExecutionCore`s need at least the central cell in their stencil buffer to produce useful outputs, they need to
     * be stepped `LATENCY` times with valid cells to "warm-up". In this context, this number of steps is called "latency".
     */
    static constexpr uindex_t latency = (grid_height + 1) * radius;

    /**
     * The number of pipeline loop bodies that have to be executed before all previous cores in the
     * pipeline produce useful values.
     * 
     * Since this core is the `position`th core in the pipeline, all previous cores produce useless
     * outputs until each of their latency steps have passed. Therefore, the whole execution
     * loop needs to be executed `WAITING_STEPS` times until this core receives useful values.
     */
    static constexpr uindex_t warmup_steps = position * latency;

    /**
     * The initial, linear counting offset for the position of the input cell.
     * 
     * To simplify their code, their is no "warm-up state" for cores; The stepping code is always fully executed.
     * Cores keep track of the input cell's position by simply counting indices, but since the the first (position - 1) * LATENCY
     * input cells aren't valid cell, these have to be left out. Therefore, cores start counting from `STARTING_POINT` and hit
     * index 0 when the first valid input cell arrives.
     */
    static constexpr uindex_t starting_point = warmup_steps == 0 ? 0 : (grid_width * grid_height) - warmup_steps;

    /**
     * Create a new execution core.
     * 
     * Due to technical problems with the dpcpp compiler, the cache can not be declared as a member of this
     * class. Instead, it has to be defined as a variable and be passed as a reference into the core object.
     */
    ExecutionCore(T (&cache)[2][grid_height][B_SIZE(radius, 1) - 1], uindex_t n_generations, Kernel kernel)
        : n_generations(n_generations), input_id(starting_point / grid_height, starting_point % grid_height), cache(cache), active_cache(0), kernel(kernel), info()
    {
        info.pipeline_position = position;
        info.may_have_sideeffects = position == 0;
        info.cell_generation = index_t(position) - index_t(pipeline_length);
    }

    /**
     * Process the next input cell, execute the stencil kernel and return the result.
     */
    T step(T input)
    {
        uindex_t &column = input_id.c;
        uindex_t &row = input_id.r;

        /**
         * Shift up every value in the stencil.
         * This operation does not touch the values in the bottom row, which will be filled
         * from the cache and the new input value later.
         */
#pragma unroll
        for (uindex_t r = 0; r < B_SIZE(radius, 1) - 1; r++)
        {
#pragma unroll
            for (uindex_t c = 0; c < B_SIZE(radius, 1); c++)
            {
                stencil[UID(c, r)] = stencil[UID(c, r + 1)];
            }
        }

        // Update the stencil buffer and cache with previous cache contents and the new input cell.
#pragma unroll
        for (uindex_t cache_c = 0; cache_c < B_SIZE(radius, 1); cache_c++)
        {
            T new_value;
            if (cache_c == B_SIZE(radius, 1) - 1)
            {
                new_value = input;
            }
            else
            {
                new_value = cache[active_cache][row][cache_c];
            }

            stencil[UID(cache_c, B_SIZE(radius, 1) - 1)] = new_value;
            if (cache_c > 0)
            {
                cache[passive_cache()][row][cache_c - 1] = new_value;
            }
        }

        // Calculate the column and row of the central cell in the stencil buffer.
        // This also handles the edge cases when the index transitions to a new row or to a new
        // generation.
        index_t kernel_c = column - radius;
        index_t kernel_r = row - radius;
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
    uindex_t passive_cache() const
    {
        return active_cache == 0 ? 1 : 0;
    }

    uindex_t n_generations;
    UID input_id;

    T (&cache)
    [2][grid_height][B_SIZE(radius, 1) - 1];
    uindex_t active_cache;

    Kernel kernel;
    Stencil2D<T, radius> stencil;
    Stencil2DInfo info;
};

/**
 * The SYCL kernel that executes the stencil kernel.
 * 
 * On the outside, this is a SYCL kernel that communicates via two pipes. One invocation is
 * initialized with a number of grid passes and the context of the stencil kernel. When executed, it consumes
 * n_passes * grid_width * grid_height values from the `in_pipe` and writes n_passes * grid_width * grid_height values to the
 * `out_pipe`, where grid_width and grid_height are the grid_width and grid_height of the grid.
 * 
 * For each grid pass, the kernel will calculate `pipeline_length` generations and only emit the cells of this last generation.
 */
template <typename T, uindex_t radius, uindex_t grid_width, uindex_t grid_height, uindex_t block_size, typename Kernel>
class ExecutionKernel
{
    static_assert(
        std::is_invocable_r<T, Kernel, Stencil2D<T, radius> const &, Stencil2DInfo const &>::
            value);
    static_assert(radius >= 1);

    uindex_t n_generations;
    uindex_t n_passes;
    Kernel kernel;

public:
    using LastCore = ExecutionCore<T, radius, grid_width, grid_height, pipeline_length, Kernel>;
    using in_pipe = cl::sycl::pipe<class in_pipe_id, T, 2 * block_size>;
    using out_pipe = cl::sycl::pipe<class out_pipe_id, T, 2 * block_size>;

    const static uindex_t n_warmup_steps = LastCore::warmup_steps;
    static_assert(grid_width * grid_height > n_warmup_steps);

    ExecutionKernel(uindex_t n_generations, uindex_t n_passes, Kernel kernel)
        : n_generations(n_generations), n_passes(n_passes), kernel(kernel) {}

    void operator()()
    {
        STENCIL_DEFINE_CORES(STENCIL_PIPELINE_LEN)

        const uindex_t n_work_steps = n_passes * grid_width * grid_height;

        for (uindex_t i = 0; i < n_warmup_steps + n_work_steps; i++)
        {
            T value;
            if (i < n_work_steps)
            {
                value = in_pipe::read();
            }

            STENCIL_STEP_CORES(STENCIL_PIPELINE_LEN)

            if (i >= n_warmup_steps)
            {
                out_pipe::write(value);
            }
        }
    }
};

} // namespace stencil