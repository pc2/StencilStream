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

namespace stencil_stream
{

template <typename T, UIndex stencil_radius, UIndex pipeline_length, UIndex output_grid_width, UIndex output_grid_height, typename TransFunc>
class ExecutionPipeline
{
public:
    static_assert(
        std::is_invocable_r<T, TransFunc, Stencil<T, stencil_radius> const &, StencilInfo const &>::
            value);
    static_assert(stencil_radius >= 1);

    const static UIndex input_grid_width = 2 * stencil_radius * pipeline_length + output_grid_width;
    const static UIndex input_grid_height = 2 * stencil_radius * pipeline_length + output_grid_height;

    using ExecutionCore = ExecutionCore<T, stencil_radius, input_grid_width, input_grid_height>;

    ExecutionPipeline(UIndex cell_generation, Index output_column_offset, Index output_row_offset, TransFunc trans_func) : cores(), trans_func(trans_func), output_column_offset(output_column_offset), output_row_offset(output_row_offset)
    {
#pragma unroll
        for (UIndex gen = 0; gen < pipeline_length; gen++)
        {
            Index intermediate_column_offset = output_column_offset - (stencil_radius * pipeline_length) - gen * stencil_radius;
            Index intermediate_row_offset = output_row_offset - (stencil_radius * pipeline_length) - gen * stencil_radius;

            cores[gen] = ExecutionCore(
                cell_generation + gen,
                intermediate_column_offset,
                intermediate_row_offset);
        }
    }

    std::optional<T> step(T input)
    {
        UIndex output_column = cores[pipeline_length - 1].get_output_column();
        UIndex output_row = cores[pipeline_length - 1].get_output_row();

        T value = input;
#pragma unroll
        for (UIndex gen = 0; gen < pipeline_length; gen++)
        {
            value = cores[gen].template step<TransFunc>(value, trans_func);
        }

        bool is_valid_output = output_column >= output_column_offset;
        is_valid_output &= output_row >= output_row_offset;
        is_valid_output &= output_column < output_column_offset + output_grid_width;
        is_valid_output &= output_row < output_row_offset + output_grid_height;

        if (is_valid_output)
        {
            return value;
        }
        else
        {
            return std::nullopt;
        }
    }

    UIndex get_total_radius() const
    {
        return stencil_radius * pipeline_length;
    }

private:
    [[intel::fpga_register]] TransFunc trans_func;
    ExecutionCore cores[pipeline_length];
    Index output_column_offset;
    Index output_row_offset;
};

} // namespace stencil_stream