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
#include "catch.hpp"
#include <CL/sycl.hpp>
#include <StencilStream/GenericID.hpp>
#include <StencilStream/Index.hpp>
#include <StencilStream/Stencil.hpp>

enum class CellStatus
{
    Normal,
    Invalid,
    Halo,
};

struct Cell
{
    stencil::index_t c;
    stencil::index_t r;
    stencil::index_t i_generation;
    CellStatus status;

    static Cell halo() { return Cell{0, 0, 0, CellStatus::Halo}; }
};

template <stencil::uindex_t radius>
class FPGATransFunc
{
public:
    Cell operator()(stencil::Stencil<Cell, radius> const &stencil) const
    {
        Cell new_cell = stencil[stencil::ID(0, 0)];

        bool is_valid = true;
#pragma unroll
        for (stencil::index_t c = -stencil::index_t(radius); c <= stencil::index_t(radius); c++)
        {
#pragma unroll
            for (stencil::index_t r = -stencil::index_t(radius); r <= stencil::index_t(radius); r++)
            {
                Cell old_cell = stencil[stencil::ID(c, r)];
                stencil::index_t cell_c = stencil.id.c + c;
                stencil::index_t cell_r = stencil.id.r + r;
                if (cell_c >= 0 && cell_r >= 0 && cell_c < stencil.grid_range.c && cell_r < stencil.grid_range.r)
                {
                    is_valid &= old_cell.c == cell_c;
                    is_valid &= old_cell.r == cell_r;
                    is_valid &= old_cell.i_generation == stencil.generation;
                    is_valid &= old_cell.status == CellStatus::Normal;
                }
                else
                {
                    is_valid &= old_cell.c == Cell::halo().c;
                    is_valid &= old_cell.r == Cell::halo().r;
                    is_valid &= old_cell.i_generation == Cell::halo().i_generation;
                    is_valid &= old_cell.status == Cell::halo().status;
                }
            }
        }

        new_cell.status = is_valid ? CellStatus::Normal : CellStatus::Invalid;
        new_cell.i_generation += 1;

        return new_cell;
    }
};

template <stencil::uindex_t radius>
class HostTransFunc
{
public:
    Cell operator()(stencil::Stencil<Cell, radius> const &stencil) const
    {
        Cell new_cell = stencil[stencil::ID(0, 0)];

#pragma unroll
        for (stencil::index_t c = -stencil::index_t(radius); c <= stencil::index_t(radius); c++)
        {
#pragma unroll
            for (stencil::index_t r = -stencil::index_t(radius); r <= stencil::index_t(radius); r++)
            {
                Cell old_cell = stencil[stencil::ID(c, r)];
                stencil::index_t cell_c = stencil.id.c + c;
                stencil::index_t cell_r = stencil.id.r + r;
                if (cell_c >= 0 && cell_r >= 0 && cell_c < stencil.grid_range.c && cell_r < stencil.grid_range.r)
                {
                    REQUIRE(old_cell.c == cell_c);
                    REQUIRE(old_cell.r == cell_r);
                    REQUIRE(old_cell.i_generation == stencil.generation);
                    REQUIRE(old_cell.status == CellStatus::Normal);
                }
                else
                {
                    REQUIRE(old_cell.c == Cell::halo().c);
                    REQUIRE(old_cell.r == Cell::halo().r);
                    REQUIRE(old_cell.i_generation == Cell::halo().i_generation);
                    REQUIRE(old_cell.status == Cell::halo().status);
                }
            }
        }
        
        new_cell.i_generation += 1;

        return new_cell;
    }
};