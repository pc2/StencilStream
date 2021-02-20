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
#include "ExecutionPipeline_pregen.hpp"

namespace stencil
{

#ifndef STENCIL_PIPELINE_LEN
#define STENCIL_PIPELINE_LEN 1
#endif

/**
 * The length of the computation pipeline in kernel replications.
 * 
 * This value defines the number of times the stencil kernel is replicated. Bigger pipeline lengths
 * lead to increased paralellity and therefore speed, but also to increased resource use and lower clock frequency.
 * 
 * The default for this value is 1 and is set by the `STENCIL_PIPELINE_LEN` macro.
 */
const UIndex pipeline_length = STENCIL_PIPELINE_LEN;

template <typename T, UIndex kernel_radius, UIndex output_grid_width, UIndex output_grid_height, typename Kernel>
class ExecutionPipeline
{
public:
    static_assert(
        std::is_invocable_r<T, Kernel, Stencil<T, kernel_radius> const &, StencilInfo const &>::
            value);
    static_assert(kernel_radius >= 1);

    ExecutionPipeline(UIndex cell_generation, UIndex output_column_offset, UIndex output_row_offset, Kernel kernel) : STENCIL_INITIALIZE_CORES(STENCIL_PIPELINE_LEN) {}

    std::optional<T> step(T input)
    {
        std::optional<T> value(input);
        STENCIL_STEP_CORES(STENCIL_PIPELINE_LEN)
        return value;
    }

private:
    STENCIL_DEFINE_CORES(STENCIL_PIPELINE_LEN);
};

} // namespace stencil