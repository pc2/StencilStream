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
#include "ExecutionCore.hpp"
#include "ExecutionKernel_pregen.hpp"
#include "Index.hpp"
#include "Stencil.hpp"
#include "StencilInfo.hpp"
#include "pipeline_length.hpp"
#include <CL/sycl/pipes.hpp>
#include <cassert>

namespace stencil
{

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
template <typename T, UIndex radius, UIndex grid_width, UIndex grid_height, UIndex block_size, typename Kernel>
class ExecutionKernel
{
    static_assert(
        std::is_invocable_r<T, Kernel, Stencil<T, radius> const &, StencilInfo const &>::
            value);
    static_assert(radius >= 1);

    UIndex n_generations;
    UIndex n_passes;
    Kernel kernel;

public:
    using LastCore = ExecutionCore<T, radius, grid_width, grid_height, pipeline_length, Kernel>;
    using in_pipe = cl::sycl::pipe<class in_pipe_id, T, 2 * block_size>;
    using out_pipe = cl::sycl::pipe<class out_pipe_id, T, 2 * block_size>;

    const static UIndex n_warmup_steps = LastCore::warmup_steps;
    static_assert(grid_width * grid_height > n_warmup_steps);

    ExecutionKernel(UIndex n_generations, UIndex n_passes, Kernel kernel)
        : n_generations(n_generations), n_passes(n_passes), kernel(kernel) {}

    void operator()()
    {
        STENCIL_DEFINE_CORES(STENCIL_PIPELINE_LEN)

        const UIndex n_work_steps = n_passes * grid_width * grid_height;

        for (UIndex i = 0; i < n_warmup_steps + n_work_steps; i++)
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