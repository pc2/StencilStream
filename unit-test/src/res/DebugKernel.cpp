/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "DebugKernel.hpp"
#include "../catch.hpp"

using namespace stencil_stream;
using Cell = DebugKernel::Cell;

Cell DebugKernel::operator()(Stencil<Cell, radius> const &stencil, StencilInfo const &info)
{
    UIndex center_column = info.center_cell_id.c;
    UIndex center_row = info.center_cell_id.r;

    for (Index c = -Index(radius); c <= Index(radius); c++)
    {
        for (Index r = -Index(radius); r <= Index(radius); r++)
        {
            REQUIRE(stencil[ID(c, r)].cell_id.c == Index(c + center_column));
            REQUIRE(stencil[ID(c, r)].cell_id.r == Index(r + center_row));
        }
    }

    Cell new_cell = stencil[ID(0, 0)];
    REQUIRE(new_cell.generation == info.cell_generation);
    new_cell.generation += 1;

    return new_cell;
}