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
#include <StencilStream/ExecutionPipeline.hpp>
#include <deque>

using namespace stencil_stream;
using namespace std;

const UIndex radius = 2;
const UIndex grid_width = 10;
const UIndex grid_height = 5;
const UIndex pipeline_length = 10;

using Kernel = DebugKernel<radius>;

TEST_CASE("ExecutionPipeline works correctly", "[ExecutionPipeline]")
{
    ExecutionPipeline<Kernel::Cell, radius, pipeline_length, grid_width, grid_height, Kernel> pipeline(0, 0, 0, Kernel(pipeline_length));

    UIndex input_grid_width = grid_width + 2 * pipeline_length * radius;
    UIndex input_grid_height = grid_height + 2 * pipeline_length * radius;
    deque<Kernel::Cell> outputs;

    for (Index c = -Index(pipeline_length * radius); c < Index(grid_width + pipeline_length * radius); c++)
    {
        for (Index r = -Index(pipeline_length * radius); r < Index(grid_height + pipeline_length * radius); r++)
        {
            Kernel::Cell cell(ID(c, r), 0);

            optional<Kernel::Cell> output = pipeline.step(cell);
            if (output.has_value())
            {
                outputs.push_back(*output);
            }
        }
    }

    REQUIRE(outputs.size() == grid_width * grid_height);

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