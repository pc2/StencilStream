/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
 * University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once
#include <CL/sycl.hpp>
#include <StencilStream/GenericID.hpp>
#include <StencilStream/Index.hpp>
#include <StencilStream/Stencil.hpp>
#include <catch2/catch_all.hpp>

enum class CellStatus {
    Normal,
    Invalid,
    Halo,
};

struct Cell {
    stencil::index_t c;
    stencil::index_t r;
    stencil::index_t i_iteration;
    stencil::index_t i_subiteration;
    CellStatus status;

    static Cell halo() { return Cell{0, 0, 0, 0, CellStatus::Halo}; }
};

struct IterationFunction {
    using Value = stencil::uindex_t;

    stencil::uindex_t operator()(stencil::uindex_t i_iteration) const { return i_iteration; }
};

template <stencil::uindex_t radius> class FPGATransFunc {
  public:
    using Cell = Cell;
    using TimeDependentValue = stencil::uindex_t;

    static constexpr stencil::uindex_t stencil_radius = radius;
    static constexpr stencil::uindex_t n_subiterations = 2;

    stencil::uindex_t get_time_dependent_value(stencil::uindex_t i_iteration) const {
        return i_iteration;
    }

    Cell operator()(stencil::Stencil<Cell, radius, TimeDependentValue> const &stencil) const {
        Cell new_cell = stencil[stencil::ID(0, 0)];

        bool is_valid = true;
#pragma unroll
        for (stencil::index_t c = -stencil::index_t(radius); c <= stencil::index_t(radius); c++) {
#pragma unroll
            for (stencil::index_t r = -stencil::index_t(radius); r <= stencil::index_t(radius);
                 r++) {
                Cell old_cell = stencil[stencil::ID(c, r)];
                stencil::index_t cell_c = stencil.id.c + c;
                stencil::index_t cell_r = stencil.id.r + r;
                if (cell_c >= 0 && cell_r >= 0 && cell_c < stencil.grid_range.c &&
                    cell_r < stencil.grid_range.r) {
                    is_valid &= old_cell.c == cell_c;
                    is_valid &= old_cell.r == cell_r;
                    is_valid &= old_cell.i_iteration == stencil.iteration;
                    is_valid &= old_cell.i_subiteration == stencil.subiteration;
                    is_valid &= old_cell.status == CellStatus::Normal;
                } else {
                    is_valid &= old_cell.c == Cell::halo().c;
                    is_valid &= old_cell.r == Cell::halo().r;
                    is_valid &= old_cell.i_iteration == Cell::halo().i_iteration;
                    is_valid &= old_cell.i_subiteration == Cell::halo().i_subiteration;
                    is_valid &= old_cell.status == Cell::halo().status;
                }
            }
        }
        is_valid &= stencil.time_dependent_value == stencil.iteration;

        new_cell.status = is_valid ? CellStatus::Normal : CellStatus::Invalid;
        if (new_cell.i_subiteration == n_subiterations - 1) {
            new_cell.i_iteration += 1;
            new_cell.i_subiteration = 0;
        } else {
            new_cell.i_subiteration++;
        }

        return new_cell;
    }
};

template <stencil::uindex_t radius> class HostTransFunc {
  public:
    using Cell = Cell;
    using TimeDependentValue = stencil::uindex_t;

    static constexpr stencil::uindex_t stencil_radius = radius;
    static constexpr stencil::uindex_t n_subiterations = 2;

    stencil::uindex_t get_time_dependent_value(stencil::uindex_t i_iteration) const {
        return i_iteration;
    }

    Cell operator()(stencil::Stencil<Cell, radius, stencil::uindex_t> const &stencil) const {
        Cell new_cell = stencil[stencil::ID(0, 0)];

        if (stencil.id.c < 0 || stencil.id.r < 0 || stencil.id.c >= stencil.grid_range.c ||
            stencil.id.r >= stencil.grid_range.r) {
            // Things may be weird in this (illegal) situation, we should not do
            // anything with effects.
            return new_cell;
        }

#pragma unroll
        for (stencil::index_t c = -stencil::index_t(radius); c <= stencil::index_t(radius); c++) {
#pragma unroll
            for (stencil::index_t r = -stencil::index_t(radius); r <= stencil::index_t(radius);
                 r++) {
                Cell old_cell = stencil[stencil::ID(c, r)];
                stencil::index_t cell_c = stencil.id.c + c;
                stencil::index_t cell_r = stencil.id.r + r;
                if (cell_c >= 0 && cell_r >= 0 && cell_c < stencil.grid_range.c &&
                    cell_r < stencil.grid_range.r) {
                    REQUIRE(old_cell.c == cell_c);
                    REQUIRE(old_cell.r == cell_r);
                    REQUIRE(old_cell.i_iteration == stencil.iteration);
                    REQUIRE(old_cell.i_subiteration == stencil.subiteration);
                    REQUIRE(old_cell.status == CellStatus::Normal);
                } else {
                    REQUIRE(old_cell.c == Cell::halo().c);
                    REQUIRE(old_cell.r == Cell::halo().r);
                    REQUIRE(old_cell.i_iteration == Cell::halo().i_iteration);
                    REQUIRE(old_cell.i_subiteration == Cell::halo().i_subiteration);
                    REQUIRE(old_cell.status == Cell::halo().status);
                }
            }
        }

        REQUIRE(stencil.time_dependent_value == stencil.iteration);

        if (new_cell.i_subiteration == n_subiterations - 1) {
            new_cell.i_iteration += 1;
            new_cell.i_subiteration = 0;
        } else {
            new_cell.i_subiteration++;
        }

        return new_cell;
    }
};