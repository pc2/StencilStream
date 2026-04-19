/*
 * Copyright © 2020-2026 Paderborn Center for Parallel Computing, Paderborn
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
#include "internal/Helpers.hpp"
#include <bit>
#include <sycl/id.hpp>
#include <sycl/range.hpp>
#include <variant>

namespace stencil {

/**
 * \brief The stencil buffer.
 *
 * The stencil buffer contains the extended Moore neighborhood of a central cell and is used by a
 * transition function to calculate the next iteration of the central cell.
 *
 * This implementation provides two ways to index the stencil: With an `ID` and a `UID`. Since `ID`
 * is signed, the column and row axes are within the range of [-radius : radius]. Therefore, (0,0)
 * points to the central cell. `UID` is unsigned and the column and row axes are within the range of
 * [0 : 2*radius + 1). Therefore, (0,0) points to the north-western corner of the stencil.
 *
 * \tparam Cell The type of cells in the stencil
 * \tparam stencil_radius The radius of the stencil, i.e. the extent of the stencil in each
 * direction from the central cell. \tparam TimeDependentValue The type of values provided by the
 * TDV system.
 */
template <typename Cell, std::size_t stencil_radius, typename TimeDependentValue = std::monostate>
    requires std::semiregular<Cell> && (stencil_radius >= 1)
class Stencil {
  public:
    /// \brief The diameter (aka width and height) of the stencil buffer.
    static constexpr std::size_t diameter = 2 * stencil_radius + 1;
    static_assert(diameter <= std::numeric_limits<int>::max());

    /**
     * \brief Create a new stencil with an uninitialized buffer.
     *
     * \param id The position of the central cell in the global grid.
     * \param grid_range The range of the underlying grid.
     * \param iteration The present iteration index of the cells in the stencil.
     * \param subiteration The present sub-iteration index of the cells in the stencil.
     * \param tdv The time-dependent value for this iteration.
     */
    Stencil(sycl::id<2> id, sycl::range<2> grid_range, std::size_t iteration,
            std::size_t subiteration, TimeDependentValue tdv)
        : id(id), iteration(iteration), subiteration(subiteration), grid_range(grid_range),
          time_dependent_value(tdv), internal() {}

    /**
     * \brief Create a new stencil with the given contents.
     *
     * \param id The position of the central cell in the global grid.
     * \param grid_range The range of the underlying grid.
     * \param iteration The present iteration index of the cells in the stencil.
     * \param subiteration The present sub-iteration index of the cells in the stencil.
     * \param tdv The time-dependent value for this iteration.
     * \param raw An array of cells, which is copied into the stencil object.
     */
    Stencil(sycl::id<2> id, sycl::range<2> grid_range, std::size_t iteration,
            std::size_t subiteration, TimeDependentValue tdv, Cell raw[diameter][diameter])
        : id(id), iteration(iteration), subiteration(subiteration), grid_range(grid_range),
          time_dependent_value(tdv), internal() {
#pragma unroll
        for (std::size_t r = 0; r < diameter; r++) {
#pragma unroll
            for (std::size_t c = 0; c < diameter; c++) {
                internal[r][c] = raw[r][c];
            }
        }
    }

    /**
     * \brief Helper returned by the signed-index subscript operator to support two-dimensional
     * indexing with signed integers.
     *
     * An expression like `stencil[-1][2]` first calls `Stencil::operator[](-1)`, which returns a
     * `StencilSubscript` holding the row index. The second `operator[](2)` on that object then
     * performs the actual cell lookup. The origin is the central cell, so row and column indices
     * range from `-stencil_radius` to `+stencil_radius` (inclusive).
     *
     * \tparam index_t A signed integer type used for indexing. Must be wide enough to represent
     * `stencil_radius`.
     */
    template <std::signed_integral index_t>
        requires(stencil_radius <= std::numeric_limits<index_t>::max())
    class StencilSubscript {
      public:
        /**
         * \brief Construct a subscript helper for the given row.
         *
         * \param stencil The stencil buffer being indexed.
         * \param r The signed row index relative to the central cell.
         */
        StencilSubscript(Stencil const &stencil, index_t r) : stencil(stencil), r(r) {}

        /**
         * \brief Access a cell at the given column offset.
         *
         * \param c The signed column index relative to the central cell.
         * \return A constant reference to the cell at row `r`, column `c`.
         */
        Cell const &operator[](index_t c) const {
            return stencil[sycl::id<2>(r + stencil_radius, c + stencil_radius)];
        }

      private:
        Stencil const &stencil;
        index_t r;
    };

    /**
     * \brief Access a row of the stencil buffer using a signed row index.
     *
     * The returned \ref StencilSubscript object can be indexed with a signed column index to
     * obtain the desired cell. Index `0` refers to the central cell's row; negative values refer
     * to rows above it and positive values to rows below it.
     *
     * \tparam index_t A signed integer type used for indexing.
     * \param r The signed row index relative to the central cell, in the range
     * `[-stencil_radius, stencil_radius]`.
     * \return A \ref StencilSubscript that resolves the column on the next `operator[]` call.
     */
    template <std::signed_integral index_t>
    StencilSubscript<index_t> operator[](index_t r) const
        requires(stencil_radius <= std::numeric_limits<index_t>::max())
    {
        return StencilSubscript<index_t>(*this, r);
    }

    /**
     * \brief Access a cell in the stencil.
     *
     * Since the indices in `id` are unsigned, the origin of this index operator is the
     * north-western corner.
     */
    Cell const &operator[](sycl::id<2> id) const { return internal[id[0]][id[1]]; }

    /**
     * \brief Access a cell in the stencil.
     *
     * Since the indices in `id` are unsigned, the origin of this index operator is the
     * north-western corner.
     */
    Cell &operator[](sycl::id<2> id) { return internal[id[0]][id[1]]; }

    /// \brief The position of the central cell in the global grid.
    const sycl::id<2> id;

    /// \brief The present iteration index of the cells in the stencil.
    const std::size_t iteration;

    /// \brief The present sub-iteration index of the cells in the stencil.
    const std::size_t subiteration;

    /// \brief The range of the underlying grid.
    const sycl::range<2> grid_range;

    /// \brief The time-dependent value for this iteration.
    const TimeDependentValue time_dependent_value;

  private:
    Cell internal[diameter][diameter];
};
} // namespace stencil
