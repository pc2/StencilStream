/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "catch.hpp"
#include "res/DebugKernel.hpp"
#include <deque>
#include <stencil/ExecutionCore.hpp>
#include <stencil/ExecutionPipeline.hpp>

using namespace stencil;
using namespace std;

const UIndex radius = 2;
const UIndex grid_width = 10;
const UIndex grid_height = 5;
const UIndex pipeline_length = 10;

TEST_CASE("ExecutionCore works correctly", "[ExecutionCore]")
{
    ExecutionCore<DebugKernel::Cell, radius, 2 * radius + grid_height> core(0, grid_width, grid_height, 0, 0);
    DebugKernel kernel;

    for (Index input_c = -Index(radius); input_c < Index(grid_width + radius); input_c++)
    {
        for (Index input_r = -Index(radius); input_r < Index(grid_height + radius); input_r++)
        {
            Index output_c = input_c - radius;
            Index output_r = input_r - radius;
            DebugKernel::Cell cell(ID(input_c, input_r), 0);

            optional<DebugKernel::Cell> output = core.template step<DebugKernel>(cell, kernel);
            if (output_c >= 0 && output_c < Index(grid_width) && output_r >= 0 && output_r < Index(grid_height))
            {
                REQUIRE(output.has_value());
                REQUIRE((*output).cell_id.c == output_c);
                REQUIRE((*output).cell_id.r == output_r);
                REQUIRE((*output).generation == 1);
            }
            else
            {
                REQUIRE(!output.has_value());
            }
        }
    }
};

TEST_CASE("ExecutionPipeline works correctly", "[ExecutionPipeline]")
{
    ExecutionPipeline<DebugKernel::Cell, radius, pipeline_length, grid_width, grid_height, DebugKernel> pipeline(0, 0, 0, DebugKernel());

    UIndex input_grid_width = grid_width + 2 * pipeline_length * radius;
    UIndex input_grid_height = grid_height + 2 * pipeline_length * radius;
    deque<DebugKernel::Cell> outputs;

    for (Index c = -Index(pipeline_length * radius); c < Index(grid_width + pipeline_length * radius); c++)
    {
        for (Index r = -Index(pipeline_length * radius); r < Index(grid_height + pipeline_length * radius); r++)
        {
            DebugKernel::Cell cell(ID(c, r), 0);

            optional<DebugKernel::Cell> output = pipeline.step(cell);
            if (output.has_value())
            {
                outputs.push_back(*output);
            }
        }
    }

    for (Index c = 0; c < grid_width; c++)
    {
        for (Index r = 0; r < grid_height; r++)
        {
            REQUIRE(outputs.front().cell_id.c == c);
            REQUIRE(outputs.front().cell_id.r == r);
            REQUIRE(outputs.front().generation == pipeline_length);
            outputs.pop_front();
        }
    }
}