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
#include <StencilStream/GenericID.hpp>
#include <StencilStream/Index.hpp>
#include <StencilStream/Stencil.hpp>
#include <StencilStream/StencilInfo.hpp>

template <stencil_stream::UIndex radius>
class DebugKernel
{
public:
    struct Cell
    {
        Cell() : cell_id(0, 0), generation(0) {}

        Cell(stencil_stream::ID cell_id, stencil_stream::UIndex generation) : cell_id(cell_id), generation(generation) {}

        Cell(Cell const &other_cell) : cell_id(other_cell.cell_id), generation(other_cell.generation) {}

        Cell &operator=(Cell const &other_cell)
        {
            cell_id = other_cell.cell_id;
            generation = other_cell.generation;
            return *this;
        }

        stencil_stream::ID cell_id;
        stencil_stream::UIndex generation;
    };

    DebugKernel(stencil_stream::UIndex pipeline_length) : pipeline_length(pipeline_length) {}

    Cell operator()(stencil_stream::Stencil<Cell, radius> const &stencil, stencil_stream::StencilInfo const &info)
    {
        stencil_stream::Index center_column = info.center_cell_id.c;
        stencil_stream::Index center_row = info.center_cell_id.r;
        stencil_stream::Index corner_id = -radius * (pipeline_length - info.cell_generation - 1);

        if (center_column >= corner_id && center_row >= corner_id)
        {
            for (stencil_stream::Index c = -stencil_stream::Index(radius); c <= stencil_stream::Index(radius); c++)
            {
                for (stencil_stream::Index r = -stencil_stream::Index(radius); r <= stencil_stream::Index(radius); r++)
                {
                    REQUIRE(stencil[stencil_stream::ID(c, r)].cell_id.c == stencil_stream::Index(c + center_column));
                    REQUIRE(stencil[stencil_stream::ID(c, r)].cell_id.r == stencil_stream::Index(r + center_row));
                }
            }
            REQUIRE(stencil[stencil_stream::ID(0, 0)].generation == info.cell_generation);
        }

        Cell new_cell = stencil[stencil_stream::ID(0, 0)];
        new_cell.generation += 1;
        return new_cell;
    }

private:
    stencil_stream::UIndex pipeline_length;
};