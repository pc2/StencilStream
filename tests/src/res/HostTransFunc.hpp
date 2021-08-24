/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
 * University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
 * associated documentation files (the “Software”), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include <StencilStream/GenericID.hpp>
#include <StencilStream/Index.hpp>
#include <StencilStream/Stencil.hpp>
#include <StencilStream/StencilInfo.hpp>

template <stencil::uindex_t radius> class HostTransFunc {
  public:
    struct Cell {
        Cell() : cell_id(0, 0), generation(0) {}

        Cell(stencil::ID cell_id, stencil::uindex_t generation)
            : cell_id(cell_id), generation(generation) {}

        Cell(Cell const &other_cell)
            : cell_id(other_cell.cell_id), generation(other_cell.generation) {}

        Cell &operator=(Cell const &other_cell) {
            cell_id = other_cell.cell_id;
            generation = other_cell.generation;
            return *this;
        }

        stencil::ID cell_id;
        stencil::uindex_t generation;
    };

    HostTransFunc(stencil::uindex_t pipeline_length) : pipeline_length(pipeline_length) {}

    Cell operator()(stencil::Stencil<Cell, radius> const &stencil,
                    stencil::StencilInfo const &info) const {
        stencil::index_t center_column = info.center_cell_id.c;
        stencil::index_t center_row = info.center_cell_id.r;
        stencil::index_t corner_id = -radius * (pipeline_length - info.cell_generation - 1);

        if (center_column >= corner_id && center_row >= corner_id) {
            for (stencil::index_t c = -stencil::index_t(radius); c <= stencil::index_t(radius);
                 c++) {
                for (stencil::index_t r = -stencil::index_t(radius); r <= stencil::index_t(radius);
                     r++) {
                    REQUIRE(stencil[stencil::ID(c, r)].cell_id.c ==
                            stencil::index_t(c + center_column));
                    REQUIRE(stencil[stencil::ID(c, r)].cell_id.r ==
                            stencil::index_t(r + center_row));
                }
            }
            REQUIRE(stencil[stencil::ID(0, 0)].generation == info.cell_generation);
        }

        Cell new_cell = stencil[stencil::ID(0, 0)];
        new_cell.generation += 1;
        return new_cell;
    }

  private:
    stencil::uindex_t pipeline_length;
};