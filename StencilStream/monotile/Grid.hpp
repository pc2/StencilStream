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
#include "../AccessorSubscript.hpp"
#include "../Concepts.hpp"
#include <cmath>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

namespace stencil {
namespace monotile {

/**
 * \brief A grid class for the monotile architecture
 *
 * This grid, which fullfils the \ref stencil::concepts::Grid "Grid" concept, contains a
 * two-dimensional buffer of cells to be used together with the \ref StencilUpdate class.
 *
 * The contents of the grid can be accessed by a host-side program using the \ref GridAccessor class
 * template. For example, one can write the contents of a `grid` object as follows:
 *
 * ```
 * Grid::GridAccessor<sycl::access::mode::read_write> accessor(grid);
 * for (std::size_t r = 0; r < grid.get_grid_height(); r++) {
 *     for (std::size_t c = 0; c < grid.get_grid_width(); c++) {
 *         accessor[r][c] = foo(r, c);
 *     }
 * }
 * ```
 *
 * Alternatively, one may write their data into a SYCL buffer and copy it into the grid using the
 * method \ref copy_from_buffer. The method \ref copy_to_buffer does the reverse: It writes the
 * contents of the grid into a SYCL buffer.
 *
 * On the device side, the data can be read or written with the help of the method templates \ref
 * submit_read and \ref submit_write. Those take a SYCL pipe as a template argument and enqueue
 * kernels that read/write the contents of the grid to/from the pipes.
 *
 * \tparam Cell The cell type to store.
 *
 * \tparam spatial_parallelism The number of cells to load and store in parallel. Has to match the
 * `spatial_parallelism` parameter for a \ref StencilUpdate instance.
 */
template <class Cell, std::size_t spatial_parallelism = 1> class Grid {
  public:
    /**
     * \brief The number of dimensions of the grid.
     *
     * May be changed in the future when other dimensions are supported.
     */
    static constexpr std::size_t dimensions = 2;

    using CellVector = Padded<std::array<Cell, spatial_parallelism>>;

    /**
     * \brief Create a new, uninitialized grid with the given dimensions.
     *
     * \param grid_height The height, or number of rows, of the new grid.
     *
     * \param grid_width The width, or number of columns, of the new grid.
     */
    Grid(std::size_t grid_height, std::size_t grid_width)
        : tile_buffer(sycl::range<2>(grid_height, int_ceil_div(grid_width, spatial_parallelism))),
          grid_range(grid_height, grid_width) {}

    /**
     * \brief Create a new, uninitialized grid with the given dimensions.
     *
     * \param range The range of the new grid. The first index will be the width and the second
     * index will be the height of the grid.
     */
    Grid(sycl::range<2> range)
        : tile_buffer(sycl::range<2>(range[0], int_ceil_div(range[1], spatial_parallelism))),
          grid_range(range) {}

    /**
     * \brief Create a new grid with the same size and contents as the given SYCL buffer.
     *
     * The contents of the buffer will be copied to the grid by the host. The SYCL buffer can later
     * be used elsewhere.
     *
     * \param buffer The buffer with the contents of the new grid.
     */
    Grid(sycl::buffer<Cell, 2> buffer)
        : tile_buffer(sycl::range<2>(buffer.get_range()[0],
                                     int_ceil_div(buffer.get_range()[1], spatial_parallelism))),
          grid_range(buffer.get_range()) {
        copy_from_buffer(buffer);
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
    Grid(Grid const &other_grid)
        : tile_buffer(other_grid.tile_buffer), grid_range(other_grid.grid_range) {}

    /**
     * \brief Create an new, uninitialized grid with the same size as the current one.
     */
    Grid make_similar() const { return Grid(grid_range); }

    /**
     * \brief Return the height, or number of rows, of the grid.
     */
    std::size_t get_grid_height() const { return grid_range[0]; }

    /**
     * \brief Return the width, or number of columns, of the grid.
     */
    std::size_t get_grid_width() const { return grid_range[1]; }

    sycl::range<2> get_grid_range() const { return grid_range; }

    /**
     * \brief An accessor for the monotile grid.
     *
     * Instances of this class provide access to a grid, so that host code can read and write the
     * contents of a grid. As such, it fullfils the \ref stencil::concepts::GridAccessor
     * "GridAccessor" concept.
     *
     * \tparam access_mode The access mode for the accessor.
     */
    template <sycl::access::mode access_mode = sycl::access::mode::read_write> class GridAccessor {
      private:
        using accessor_t = sycl::host_accessor<CellVector, 2, access_mode>;

      public:
        /**
         * \brief The number of dimensions of the underlying grid.
         */
        static constexpr std::size_t dimensions = Grid::dimensions;

        /**
         * \brief Create a new accessor to the given grid.
         */
        GridAccessor(Grid &grid) : ac(grid.tile_buffer) {}

        /**
         * \brief Shorthand for the used subscript type.
         */
        using BaseSubscript = AccessorSubscript<Cell, GridAccessor, access_mode>;

        /**
         * \brief Access/Dereference the first dimension.
         *
         * This subscript operator is the first subscript in an expression like
         * `accessor[i_row][i_column]`. It will return a \ref BaseSubscript object that handles
         * subsequent dimensions.
         */
        BaseSubscript operator[](std::size_t i) { return BaseSubscript(*this, i); }

        /**
         * \brief Access a cell of the grid.
         *
         * \param id The index of the accessed cell. The first index is the row index, the second
         * one is the column index.
         *
         * \returns A constant reference to the indexed cell.
         */
        Cell const &operator[](sycl::id<2> id)
            requires(access_mode == sycl::access::mode::read)
        {
            return ac[id[0]][id[1] / spatial_parallelism].value[id[1] % spatial_parallelism];
        }

        /**
         * \brief Access a cell of the grid.
         *
         * \param id The index of the accessed cell. The first index is the row index, the second
         * one is the column index. \returns A reference to the indexed cell.
         */
        Cell &operator[](sycl::id<2> id)
            requires(access_mode != sycl::access::mode::read)
        {
            return ac[id[0]][id[1] / spatial_parallelism].value[id[1] % spatial_parallelism];
        }

      private:
        accessor_t ac;
    };

    /**
     * \brief Copy the contents of the SYCL buffer into the grid.
     *
     * The SYCL buffer will be accessed read-only one the host; It may be used elsewhere too. The
     * buffer however has to have the same size as the grid, otherwise a \ref std::range_error is
     * thrown.
     *
     * \param input_buffer The buffer to copy the data from.
     * \throws std::range_error The size of the buffer does not match the grid.
     */
    void copy_from_buffer(sycl::buffer<Cell, 2> input_buffer) {
        assert(tile_buffer.get_range()[1] == int_ceil_div(grid_range[1], spatial_parallelism));
        if (input_buffer.get_range() != grid_range) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        sycl::host_accessor in_ac(input_buffer, sycl::read_only);
        sycl::host_accessor tile_ac(tile_buffer, sycl::read_write);
        for (std::size_t r = 0; r < grid_range[0]; r++) {
            for (std::size_t c = 0; c < grid_range[1]; c++) {
                tile_ac[r][c / spatial_parallelism].value[c % spatial_parallelism] = in_ac[r][c];
            }
        }
    }

    /**
     * \brief Copy the contents of the grid into the SYCL buffer.
     *
     * The contents of the SYCL buffer will be overwritten on the host. The buffer also has to have
     * the same size as the grid, otherwise a \ref std::range_error is thrown.
     *
     * \param output_buffer The buffer to copy the data to.
     * \throws std::range_error The size of the buffer does not match the grid.
     */
    void copy_to_buffer(sycl::buffer<Cell, 2> output_buffer) {
        assert(tile_buffer.get_range()[1] == int_ceil_div(grid_range[1], spatial_parallelism));
        if (output_buffer.get_range() != grid_range) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        sycl::host_accessor tile_ac(tile_buffer, sycl::read_only);
        sycl::host_accessor out_ac(output_buffer, sycl::write_only);
        for (std::size_t r = 0; r < grid_range[0]; r++) {
            for (std::size_t c = 0; c < grid_range[1]; c++) {
                out_ac[r][c] = tile_ac[r][c / spatial_parallelism].value[c % spatial_parallelism];
            }
        }
    }

    sycl::buffer<CellVector, 2> get_internal() const { return tile_buffer; }

  private:
    sycl::buffer<CellVector, 2> tile_buffer;
    sycl::range<2> grid_range;
};
} // namespace monotile
} // namespace stencil