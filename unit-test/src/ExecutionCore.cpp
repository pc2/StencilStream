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
#include <StencilStream/ExecutionCore.hpp>

using namespace stencil_stream;
using namespace std;

const UIndex radius = 2;
const UIndex grid_width = 10;
const UIndex grid_height = 5;

using Kernel = DebugKernel<radius>;

TEST_CASE("ExecutionCore works correctly", "[ExecutionCore]")
{
    ExecutionCore<Kernel::Cell, radius, 2 * radius + grid_width, 2 * radius + grid_height> core(0, -radius, -radius);
    Kernel kernel(1);

    for (Index input_c = -Index(radius); input_c < Index(grid_width + radius); input_c++)
    {
        for (Index input_r = -Index(radius); input_r < Index(grid_height + radius); input_r++)
        {
            Kernel::Cell cell(ID(input_c, input_r), 0);

            Kernel::Cell output = core.template step<Kernel>(cell, kernel);

            if (input_c >= Index(2 * radius + 1) && input_r >= Index(2 * radius + 1))
            {
                REQUIRE(output.cell_id.c == input_c - radius);
                REQUIRE(output.cell_id.r == input_r - radius);
                REQUIRE(output.generation == 1);
            }
        }
    }
};
