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
#include "Index.hpp"
#include <CL/sycl.hpp>

namespace stencil {
/**
 * @brief A helper class to support the double-subscript idiom for
 * \ref stencil::concepts::GridAccessor "GridAccessors"
 *
 * This class is used to support expressions like `accessor[i_column][i_row]`. Each subscript except
 * the last one creates an object of this class, which stores the index of the subscript as well as
 * all previous indices to form a so-called "prefix." When the last subscript is called, the index
 * is added to the prefix to yield the complete, multi-dimensional index. Then, the last subscript
 * object returns a reference to the requested cell.
 *
 * \tparam Cell The cell type to support
 *
 * \tparam Accessor accessor to direct to. It must fulfill \ref
 * stencil::concepts::GridAccessor, but since this class is also part of the concept's definition,
 * we can't check it here.
 *
 * \tparam access_mode The access mode of the accessor. Used to enable and disable different \ref
 * operator[] implementations
 *
 * \tparam current_subdim The current subdim. Decides whether the next call to \ref
 * operator[] returns another \ref AccessorSubscript or the final cell reference.
 */
template <typename Cell, typename Accessor, sycl::access::mode access_mode,
          uindex_t current_subdim = 0>
class AccessorSubscript {
  public:
    /// \brief The number of dimensions in the accessed grid.
    static constexpr uindex_t dimensions = Accessor::dimensions;

    /**
     * \brief Instantiate a new accessor subscript object with the given index as the prefix.
     *
     * This accessor is meant to be used by the `operator[]` implementation of the accessor to
     * construct the first subscript object. It is therefore only available iff the current
     * subdimention is zero.
     *
     * \param ac The accessor to redirect to.
     * \param i The index of the previous subscript. It will be used as a prefix.
     */
    AccessorSubscript(Accessor &ac, uindex_t i)
        requires(current_subdim == 0)
        : ac(ac), id_prefix() {
        id_prefix[current_subdim] = i;
    }

    /**
     * \brief Instantiate a new subscript object with the given prefix and subscript index.
     *
     * This constructor will take the given prefix, add the new index to it at the current position,
     * and use the resulting prefix.
     *
     * \param ac The accessor to redirect to.
     * \param id_prefix The previous prefix
     * \param i The new index to add to the prefix.
     */
    AccessorSubscript(Accessor &ac, sycl::id<dimensions> id_prefix, uindex_t i)
        : ac(ac), id_prefix(id_prefix) {
        id_prefix[current_subdim] = i;
    }

    /**
     * \brief Access the next dimension's accessor subscript.
     *
     * The resulting subscript object will contain the current prefix together with the given index.
     * This implementation is only available to intermediate dimensions. For the last dimension, the
     * resulting ID will be evaluated and returned.
     *
     * \param i The next dimension's index.
     *
     * \return A subscript object for the following dimensions.
     */
    AccessorSubscript<Cell, Accessor, access_mode, current_subdim + 1> operator[](uindex_t i)
        requires(current_subdim < dimensions - 2)
    {
        return AccessorSubscript(ac, id_prefix, i);
    }

    /**
     * \brief Access the cell.
     *
     * This will add the given index to the prefix and use the underlying accessor to provide a
     * reference to the cell. This is the implementation for read-only accessors as it returns a
     * constant reference.
     *
     * \param i The final dimension's index.
     *
     * \return A constant reference to the indexed cell.
     */
    Cell const &operator[](uindex_t i)
        requires(current_subdim == dimensions - 2 && access_mode == sycl::access::mode::read)
    {
        sycl::id<dimensions> id = id_prefix;
        id[current_subdim + 1] = i;
        return ac[id];
    }

    /**
     * \brief Access the cell.
     *
     * This will add the given index to the prefix and use the underlying accessor to provide a
     * reference to the cell. This is the implementation for accessors that may alter the grid.
     *
     * \param i The final dimension's index.
     *
     * \return A reference to the indexed cell.
     */
    Cell &operator[](uindex_t i)
        requires(current_subdim == dimensions - 2 && access_mode != sycl::access::mode::read)
    {
        sycl::id<dimensions> id = id_prefix;
        id[current_subdim + 1] = i;
        return ac[id];
    }

  private:
    Accessor &ac;
    sycl::id<dimensions> id_prefix;
};
} // namespace stencil