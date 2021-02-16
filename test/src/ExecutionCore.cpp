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
#include <ExecutionCore.hpp>

using namespace stencil;

const UIndex radius = 2;
const UIndex grid_width = 10;
const UIndex grid_height = 5;

ID kernel(Stencil<ID, radius> const &stencil, StencilInfo const &info)
{
    UIndex center_column = info.center_cell_id.c;
    UIndex center_row = info.center_cell_id.r;

    bool is_valid = true;
    for (Index c = -Index(radius); c <= Index(radius); c++)
    {
        for (Index r = -Index(radius); r <= Index(radius); r++)
        {
            REQUIRE(stencil[ID(c, r)].c == Index(c + center_column));
            REQUIRE(stencil[ID(c, r)].r == Index(r + center_row));
        }
    }

    return stencil[ID(0, 0)];
}

TEST_CASE("ExecutionCore works correctly", "[ExecutionCore]")
{
    ID cache[2][2*radius + grid_height][Stencil<ID, radius>::diameter() - 1];

    ExecutionCore<ID, radius, grid_width, grid_height, decltype(&kernel)> core(cache, 0, 0, 0, &kernel);

    for (Index input_c = -Index(radius); input_c < Index(grid_width + radius); input_c++)
    {
        for (Index input_r = -Index(radius); input_r < Index(grid_height + radius); input_r++)
        {
            Index output_c = input_c - radius;
            Index output_r = input_r - radius;
            std::optional<ID> output = core.step(ID(input_c, input_r));
            if (output_c >= 0 && output_c < Index(grid_width) && output_r >= 0 && output_r < Index(grid_height))
            {
                REQUIRE(output.has_value());
                REQUIRE((*output).c == output_c);
                REQUIRE((*output).r == output_r); 
            }
            else
            {
                REQUIRE(!output.has_value());
            }
        }
    }
};