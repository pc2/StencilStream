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
#include <stencil/GenericID.hpp>
#include <stencil/Index.hpp>
#include <stencil/Stencil.hpp>
#include <stencil/StencilInfo.hpp>

class DebugKernel
{
public:
    struct Cell
    {
        Cell() : cell_id(0, 0), generation(0) {}

        Cell(stencil::ID cell_id, stencil::UIndex generation) : cell_id(cell_id), generation(generation) {}

        Cell(Cell const &other_cell) : cell_id(other_cell.cell_id), generation(other_cell.generation) {}

        Cell &operator=(Cell const &other_cell)
        {
            cell_id = other_cell.cell_id;
            generation = other_cell.generation;
            return *this;
        }

        stencil::ID cell_id;
        stencil::UIndex generation;
    };

    const static stencil::UIndex radius = 2;

    Cell operator()(stencil::Stencil<Cell, radius> const &stencil, stencil::StencilInfo const &info);
};