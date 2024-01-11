/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "GenericID.hpp"
#include "Helpers.hpp"
#include "Index.hpp"
#include <bit>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <variant>

namespace stencil {

/**
 * \brief The stencil buffer.
 *
 * The stencil buffer contains the extended Moore neighborhood of a central cell and is used by the
 * transition function to calculate the next generation of the central cell.
 *
 * This implementation provides two ways to index the stencil: With an `ID` and a `UID`. Since `ID`
 * is signed, the column and row axes are within the range of [-radius : radius]. Therefore, (0,0)
 * points to the central cell. `UID` is unsigned and the column and row axes are within the range of
 * [0 : 2*radius + 1). Therefore, (0,0) points to the north-western corner of the stencil.
 */
template <typename Cell, uindex_t stencil_radius, typename TimeDependentValue = std::monostate>
    requires std::semiregular<Cell> && (stencil_radius >= 1)
class Stencil {
  public:
    /**
     * \brief The diameter (aka width and height) of the stencil buffer.
     */
    static constexpr uindex_t diameter = 2 * stencil_radius + 1;

    static constexpr unsigned long bits_stencil = std::bit_width(diameter);
    using index_stencil_t = ac_int<bits_stencil, true>;
    using uindex_stencil_t = ac_int<bits_stencil, false>;

    using StencilID = GenericID<index_stencil_t>;
    using StencilUID = GenericID<uindex_stencil_t>;

    /**
     * \brief Create a new stencil with an uninitialized buffer.
     *
     * \param id The position of the central cell in the global grid.
     * \param generation The present generation index of the central cell.
     * \param i_processing_element The index of the processing element that calls the transition
     * function.
     * \param grid_range The range of the stencil's grid.
     */
    Stencil(ID id, UID grid_range, uindex_t generation, uindex_t subgeneration,
            uindex_t i_processing_element, TimeDependentValue tdv)
        : id(id), generation(generation), subgeneration(subgeneration),
          i_processing_element(i_processing_element), grid_range(grid_range),
          time_dependent_value(tdv), internal() {}

    /**
     * \brief Create a new stencil from the raw buffer.
     *
     * \param id The position of the central cell in the global grid.
     * \param generation The present generation index of the central cell.
     * \param i_processing_element The index of the processing element that calls the transition
     * function.
     * \param raw A raw array containing cells.
     * \param grid_range The range of the stencil's grid.
     */
    Stencil(ID id, UID grid_range, uindex_t generation, uindex_t subgeneration,
            uindex_t i_processing_element, TimeDependentValue tdv, Cell raw[diameter][diameter])
        : id(id), generation(generation), subgeneration(subgeneration),
          i_processing_element(i_processing_element), grid_range(grid_range),
          time_dependent_value(tdv), internal() {
#pragma unroll
        for (uindex_t c = 0; c < diameter; c++) {
#pragma unroll
            for (uindex_t r = 0; r < diameter; r++) {
                internal[c][r] = raw[c][r];
            }
        }
    }

    /**
     * \brief Access a cell in the stencil.
     *
     * Since the indices in `id` are signed, the origin of this index operator is the central cell.
     */
    Cell const &operator[](ID id) const {
        return internal[id.c + stencil_radius][id.r + stencil_radius];
    }

    /**
     * \brief Access a cell in the stencil.
     *
     * Since the indices in `id` are signed, the origin of this index operator is the central cell.
     */
    Cell &operator[](ID id) { return internal[id.c + stencil_radius][id.r + stencil_radius]; }

    /**
     * \brief Access a cell in the stencil.
     *
     * Since the indices in `id` are signed, the origin of this index operator is the central cell.
     */
    Cell const &operator[](StencilID id) const {
        return internal[id.c + index_stencil_t(stencil_radius)]
                       [id.r + index_stencil_t(stencil_radius)];
    }

    /**
     * \brief Access a cell in the stencil.
     *
     * Since the indices in `id` are signed, the origin of this index operator is the central cell.
     */
    Cell &operator[](StencilID id) {
        return internal[id.c + index_stencil_t(stencil_radius)]
                       [id.r + index_stencil_t(stencil_radius)];
    }

    /**
     * \brief Access a cell in the stencil.
     *
     * Since the indices in `id` are unsigned, the origin of this index operator is the
     * north-western corner.
     */
    Cell const &operator[](UID id) const { return internal[id.c][id.r]; }

    /**
     * \brief Access a cell in the stencil.
     *
     * Since the indices in `id` are unsigned, the origin of this index operator is the
     * north-western corner.
     */
    Cell &operator[](UID id) { return internal[id.c][id.r]; }

    /**
     * \brief Access a cell in the stencil.
     *
     * Since the indices in `id` are unsigned, the origin of this index operator is the
     * north-western corner.
     */
    Cell const &operator[](StencilUID id) const { return internal[id.c][id.r]; }

    /**
     * \brief Access a cell in the stencil.
     *
     * Since the indices in `id` are unsigned, the origin of this index operator is the
     * north-western corner.
     */
    Cell &operator[](StencilUID id) { return internal[id.c][id.r]; }

    /**
     * \brief The position of the central cell in the global grid.
     */
    const ID id;

    /**
     * \brief The present generation index of the central cell.
     */
    const uindex_t generation;

    const uindex_t subgeneration;

    /**
     * \brief The index of the processing element that calls the transition function.
     *
     * The processing element index is added to the generation index of the input tile to get the
     * generation index of this stencil. Under the assumption that the pipeline was always fully
     * executed, it equals to `generation % n_processing_elements`. Since it is hard coded in the
     * final design, it can be used to alternate between two different data paths: When you write
     * something like this
     * ```
     * if (stencil.i_processing_element & 0b1 == 0)
     * {
     *   return foo(stencil);
     * }
     * else
     * {
     *   return bar(stencil);
     * }
     * ```
     * `foo` will only be synthesized for even processing elements, and `bar` will only be
     * synthesized for odd processing elements.
     */
    const uindex_t i_processing_element;

    /**
     * \brief The number of columns and rows of the grid.
     */
    const UID grid_range;

    const TimeDependentValue time_dependent_value;

  private:
    Cell internal[diameter][diameter];
};
} // namespace stencil
