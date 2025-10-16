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
#include <StencilStream/Stencil.hpp>
#include <catch2/catch_all.hpp>
#include <sycl/sycl.hpp>

template <typename T> struct cell_members;

enum class CellStatus {
    Normal,
    Invalid,
    Halo,
};

struct Cell {
    int r;
    int c;
    int i_iteration;
    int i_subiteration;
    CellStatus status;

    static Cell halo() { return Cell{0, 0, 0, 0, CellStatus::Halo}; }
};

template <> struct cell_members<Cell> {
    static constexpr auto fields = std::make_tuple(&Cell::r, &Cell::c, &Cell::i_iteration,
                                                   &Cell::i_subiteration, &Cell::status);
};

struct IterationFunction {
    using Value = std::size_t;

    std::size_t operator()(std::size_t i_iteration) const { return i_iteration; }
};

template <std::size_t radius> class FPGATransFunc {
  public:
    using Cell = Cell;
    using TimeDependentValue = std::size_t;

    static constexpr std::size_t stencil_radius = radius;
    static constexpr std::size_t n_subiterations = 2;

    std::size_t get_time_dependent_value(std::size_t i_iteration) const { return i_iteration; }

    Cell operator()(stencil::Stencil<Cell, radius, TimeDependentValue> const &stencil) const {
        Cell new_cell = stencil[0][0];

        bool is_valid = true;
#pragma unroll
        for (int r = -int(radius); r <= int(radius); r++) {
#pragma unroll
            for (int c = -int(radius); c <= int(radius); c++) {
                Cell old_cell = stencil[r][c];
                int cell_r = stencil.id[0] + r;
                int cell_c = stencil.id[1] + c;
                if (cell_r >= 0 && cell_c >= 0 && cell_r < stencil.grid_range[0] &&
                    cell_c < stencil.grid_range[1]) {
                    is_valid &= old_cell.r == cell_r;
                    is_valid &= old_cell.c == cell_c;
                    is_valid &= old_cell.i_iteration == stencil.iteration;
                    is_valid &= old_cell.i_subiteration == stencil.subiteration;
                    is_valid &= old_cell.status == CellStatus::Normal;
                } else {
                    is_valid &= old_cell.r == Cell::halo().r;
                    is_valid &= old_cell.c == Cell::halo().c;
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

template <std::size_t radius> class HostTransFunc {
  public:
    using Cell = Cell;
    using TimeDependentValue = std::size_t;

    static constexpr std::size_t stencil_radius = radius;
    static constexpr std::size_t n_subiterations = 2;

    std::size_t get_time_dependent_value(std::size_t i_iteration) const { return i_iteration; }

    Cell operator()(stencil::Stencil<Cell, radius, std::size_t> const &stencil) const {
        Cell new_cell = stencil[0][0];

        if (stencil.id[0] >= stencil.grid_range[0] || stencil.id[1] >= stencil.grid_range[1]) {
            // Things may be weird in this (illegal) situation, we should not do
            // anything with side-effects.
            return new_cell;
        }

        for (int r = -int(radius); r <= int(radius); r++) {
            for (int c = -int(radius); c <= int(radius); c++) {
                Cell old_cell = stencil[r][c];
                int cell_r = stencil.id[0] + r;
                int cell_c = stencil.id[1] + c;
                if (cell_r >= 0 && cell_c >= 0 && cell_r < stencil.grid_range[0] &&
                    cell_c < stencil.grid_range[1]) {
                    REQUIRE(old_cell.r == cell_r);
                    REQUIRE(old_cell.c == cell_c);
                    REQUIRE(old_cell.i_iteration == stencil.iteration);
                    REQUIRE(old_cell.i_subiteration == stencil.subiteration);
                    REQUIRE(old_cell.status == CellStatus::Normal);
                } else {
                    REQUIRE(old_cell.r == Cell::halo().r);
                    REQUIRE(old_cell.c == Cell::halo().c);
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