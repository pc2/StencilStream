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
#include "../Index.hpp"
#include <memory>

namespace stencil {
namespace cpu {

/**
 * \brief A grid class for the CPU backend
 *
 * This grid, which fullfils the \ref stencil::concepts::Grid "Grid" concept, contains a
 * two-dimensional buffer of cells to be used together with the \ref StencilUpdate class.
 *
 * The contents of the grid can be accessed by a host-side program using the \ref GridAccessor class
 * template. For example, one can write the contents of a `grid` object as follows:
 *
 * ```
 * Grid::GridAccessor<sycl::access::mode::read_write> accessor(grid);
 * for (uindex_t c = 0; c < grid.get_grid_width(); c++) {
 *     for (uindex_t r = 0; r < grid.get_grid_height(); r++) {
 *         accessor[c][r] = foo(c, r);
 *     }
 * }
 * ```
 *
 * Alternatively, one may write their data into a SYCL buffer and copy it into the grid using the
 * method \ref copy_from_buffer. The method \ref copy_to_buffer does the reverse: It writes the
 * contents of the grid into a SYCL buffer.
 *
 * \tparam Cell The cell type to store.
 */
template <typename Cell> class Grid {
  public:
    /**
     * \brief The number of dimensions of the grid.
     *
     * May be changed in the future when other dimensions are supported.
     */
    static constexpr uindex_t dimensions = 2;

    /**
     * \brief Create a new, uninitialized grid with the given dimensions.
     *
     * \param c The width, or number of columns, of the new grid.
     *
     * \param r The height, or number of rows, of the new grid.
     */
    Grid(uindex_t c, uindex_t r) : buffer(sycl::range<2>(c, r)) {}

    /**
     * \brief Create a new, uninitialized grid with the given dimensions.
     *
     * \param range The range of the new grid. The first index will be the width and the second
     * index will be the height of the grid.
     */
    Grid(sycl::range<2> range) : buffer(range) {}

    /**
     * \brief Create a new grid with the same size and contents as the given SYCL buffer.
     *
     * The contents of the buffer will be copied to the grid by the host. The SYCL buffer can later
     * be used elsewhere.
     *
     * \param other_buffer The buffer with the contents of the new grid.
     */
    Grid(sycl::buffer<Cell, 2> other_buffer) : buffer(other_buffer.get_range()) {
        copy_from_buffer(other_buffer);
    }

    /**
     * \brief Create a new reference to the given grid.
     *
     * The newly created grid object will point to the same underlying data as the referenced grid.
     * Changes made via the newly created grid object will also be visible to the old grid object,
     * and vice-versa.
     *
     * \param other_grid The other grid the new grid should reference.
     */
    Grid(Grid const &other_grid) : buffer(other_grid.buffer) {}

    /**
     * \brief Copy the contents of the SYCL buffer into the grid.
     *
     * The SYCL buffer will be accessed read-only one the host; It may be used elsewhere too. The
     * buffer however has to have the same size as the grid, otherwise a \ref std::range_error is
     * thrown.
     *
     * \param other_buffer The buffer to copy the data from.
     * \throws std::range_error The size of the buffer does not match the grid.
     */
    void copy_from_buffer(sycl::buffer<Cell, 2> other_buffer) {
        if (buffer.get_range() != other_buffer.get_range()) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }
        sycl::host_accessor buffer_ac(buffer, sycl::write_only);
        sycl::host_accessor other_ac(other_buffer, sycl::read_only);
        std::memcpy(buffer_ac.get_pointer(), other_ac.get_pointer(), buffer_ac.byte_size());
    }

    /**
     * \brief Copy the contents of the grid into the SYCL buffer.
     *
     * The contents of the SYCL buffer will be overwritten on the host. The buffer also has to have
     * the same size as the grid, otherwise a \ref std::range_error is thrown.
     *
     * \param other_buffer The buffer to copy the data to.
     * \throws std::range_error The size of the buffer does not match the grid.
     */
    void copy_to_buffer(sycl::buffer<Cell, 2> other_buffer) {
        if (buffer.get_range() != other_buffer.get_range()) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }
        sycl::host_accessor buffer_ac(buffer, sycl::read_only);
        sycl::host_accessor other_ac(other_buffer, sycl::write_only);
        std::memcpy(other_ac.get_pointer(), buffer_ac.get_pointer(), buffer_ac.byte_size());
    }

    /**
     * \brief An accessor for the grid.
     *
     * Instances of this class provide access to a grid, so that host code can read and write the
     * contents of a grid. As such, it fullfils the \ref stencil::concepts::GridAccessor
     * "GridAccessor" concept.
     *
     * \tparam access_mode The access mode for the accessor.
     */
    template <sycl::access::mode access_mode = sycl::access::mode::read_write>
    class GridAccessor : public sycl::host_accessor<Cell, Grid::dimensions, access_mode> {
      public:
        /**
         * \brief Create a new accessor to the given grid.
         */
        GridAccessor(Grid &grid)
            : sycl::host_accessor<Cell, Grid::dimensions, access_mode>(grid.buffer) {}
    };

    /**
     * \brief Return the width, or number of columns, of the grid.
     */
    uindex_t get_grid_width() const { return buffer.get_range()[0]; }

    /**
     * \brief Return the height, or number of rows, of the grid.
     */
    uindex_t get_grid_height() const { return buffer.get_range()[1]; }

    /**
     * \brief Create an new, uninitialized grid with the same size as the current one.
     */
    Grid make_similar() const { return Grid(get_grid_width(), get_grid_height()); }

    sycl::buffer<Cell, 2> &get_buffer() { return buffer; }

  private:
    sycl::buffer<Cell, 2> buffer;
};
} // namespace cpu
} // namespace stencil