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
#include "../GenericID.hpp"
#include <memory>
#include <numeric>
#include <vector>

namespace stencil {
namespace tiling {

/**
 * \brief A grid class for the tiling architecture
 *
 * This grid, which fullfils the \ref stencil::concepts::Grid "Grid" concept, contains a
 * two-dimensional buffer of cells to be used together with the \ref StencilUpdate class.
 *
 * Since this class template requires multiple template arguments that must match the used \ref
 * StencilUpdate instance, it is advised to use the \ref StencilUpdate::GridImpl shorthand instead
 * of re-applying the same arguments to the grid class template.
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
 * On the device side, the data can be read or written with the help of the method templates \ref
 * submit_read and \ref submit_write. Those take a SYCL pipe as a template argument and enqueue
 * kernels that read/write the contents of the grid to/from the pipes.
 *
 * \tparam Cell The cell type to store.
 *
 * \tparam tile_width The width of a grid tile. This has to match the tile width of the used \ref
 * StencilUpdate class instance.
 *
 * \tparam tile_height The height of a grid tile. This has to match the tile height of the used \ref
 * StencilUpdate class instance.
 *
 * \tparam halo_radius The halo radius required for input tiles. This has to be the number of PEs in
 * a \ref StencilUpdate times the stencil radius of the implemented transition function.
 */
template <typename Cell, uindex_t tile_width = 1024, uindex_t tile_height = 1024,
          uindex_t halo_radius = 1>
class Grid {
    static_assert(2 * halo_radius < tile_height && 2 * halo_radius < tile_width);

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
     * \param grid_width The width, or number of columns, of the new grid.
     *
     * \param grid_height The height, or number of rows, of the new grid.
     */
    Grid(uindex_t grid_width, uindex_t grid_height)
        : grid_buffer(sycl::range<2>(grid_width, grid_height)) {}

    /**
     * \brief Create a new, uninitialized grid with the given dimensions.
     *
     * \param range The range of the new grid. The first index will be the width and the second
     * index will be the height of the grid.
     */
    Grid(sycl::range<2> range) : grid_buffer(range) {}

    /**
     * \brief Create a new grid with the same size and contents as the given SYCL buffer.
     *
     * The contents of the buffer will be copied to the grid by the host. The SYCL buffer can later
     * be used elsewhere.
     *
     * \param input_buffer The buffer with the contents of the new grid.
     */
    Grid(sycl::buffer<Cell, 2> input_buffer) : grid_buffer(input_buffer.get_range()) {
        copy_from_buffer(input_buffer);
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
    Grid(Grid const &other_grid) : grid_buffer(other_grid.grid_buffer) {}

    /**
     * \brief An accessor for the monotile grid.
     *
     * Instances of this class provide access to a grid, so that host code can read and write the
     * contents of a grid. As such, it fullfils the \ref stencil::concepts::GridAccessor
     * "GridAccessor" concept.
     *
     * \tparam access_mode The access mode for the accessor.
     */
    template <sycl::access::mode access_mode> class GridAccessor {
      public:
        /**
         * \brief The number of dimensions of the underlying grid.
         */
        static constexpr uindex_t dimensions = Grid::dimensions;

        /**
         * \brief Create a new accessor to the given grid.
         */
        GridAccessor(Grid &grid) : accessor(grid.grid_buffer) {}

        /**
         * \brief Shorthand for the used subscript type.
         */
        using BaseSubscript = AccessorSubscript<Cell, GridAccessor, access_mode>;

        /**
         * \brief Access/Dereference the first dimension.
         *
         * This subscript operator is the first subscript in an expression like
         * `accessor[i_column][i_row]`. It will return a \ref BaseSubscript object that handles
         * subsequent dimensions.
         */
        BaseSubscript operator[](uindex_t i) { return BaseSubscript(*this, i); }

        /**
         * \brief Access a cell of the grid.
         *
         * \param id The index of the accessed cell. The first index is the column index, the second
         * one is the row index. \returns A constant reference to the indexed cell.
         */
        Cell const &operator[](sycl::id<2> id)
            requires(access_mode == sycl::access::mode::read)
        {
            return accessor[id];
        }

        /**
         * \brief Access a cell of the grid.
         *
         * \param id The index of the accessed cell. The first index is the column index, the second
         * one is the row index. \returns A reference to the indexed cell.
         */
        Cell &operator[](sycl::id<2> id)
            requires(access_mode != sycl::access::mode::read)
        {
            return accessor[id];
        }

      private:
        sycl::host_accessor<Cell, 2, access_mode> accessor;
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
        if (input_buffer.get_range() != grid_buffer.get_range()) {
            throw std::out_of_range("The target buffer has not the same size as the grid");
        }

        sycl::host_accessor grid_ac{grid_buffer, sycl::write_only};
        sycl::host_accessor input_ac{input_buffer, sycl::read_only};
        for (uindex_t c = 0; c < get_grid_width(); c++) {
            for (uindex_t r = 0; r < get_grid_height(); r++) {
                grid_ac[c][r] = input_ac[c][r];
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
        if (output_buffer.get_range() != grid_buffer.get_range()) {
            throw std::out_of_range("The target buffer has not the same size as the grid");
        }

        sycl::host_accessor grid_ac{grid_buffer, sycl::read_only};
        sycl::host_accessor output_ac{output_buffer, sycl::write_only};
        for (uindex_t c = 0; c < get_grid_width(); c++) {
            for (uindex_t r = 0; r < get_grid_height(); r++) {
                output_ac[c][r] = grid_ac[c][r];
            }
        }
    }

    /**
     * \brief Create an new, uninitialized grid with the same size as the current one.
     */
    Grid make_similar() const { return Grid(grid_buffer.get_range()); }

    /**
     * \brief Return the width, or number of columns, of the grid.
     */
    uindex_t get_grid_width() const { return grid_buffer.get_range()[0]; }

    /**
     * \brief Return the height, or number of rows, of the grid.
     */
    uindex_t get_grid_height() const { return grid_buffer.get_range()[1]; }

    /**
     * \brief Return the range of (central) tiles of the grid.
     *
     * This is not the range of a single tile nor is the range of the grid. It is the range of valid
     * arguments for \ref submit_read and \ref submit_write. For example, if the grid is 60 by 60
     * cells in size and a tile is 32 by 32 cells in size, the tile range would be 2 by 2 tiles.
     *
     * \return The range of tiles of the grid.
     */
    GenericID<uindex_t> get_tile_range() const {
        return GenericID<uindex_t>(std::ceil(float(get_grid_width()) / float(tile_width)),
                                   std::ceil(float(get_grid_height()) / float(tile_height)));
    }

    /**
     * \brief Submit a kernel that sends a tile of the grid into a pipe.
     *
     * The submitted kernel will send the contents of a tile, along with the required halo, into a
     * pipe in column-major order. This means that the last index (which denotes the row) will
     * change the quickest. The method returns the event of the launched kernel immediately.
     *
     * The tile column and row indices denote the indices of the *tile*, not the cells. The starting
     * column index will be `tile_c * tile_width - halo_radius` and the end index will be `(tile_c +
     * 1) * tile_width + halo_radius`. The start and end indices for the rows are analogous.
     *
     * This method is explicitly part of the user-facing API: You are allowed and encouraged to use
     * this method to feed custom kernels.
     *
     * \tparam in_pipe The pipe the data is sent into.
     *
     * \param queue The queue to submit the kernel to.
     *
     * \param tile_c The column index of the tile to read.
     *
     * \param tile_r The row index of the tile to read.
     *
     * \param halo_value The value to present for cells outside of the grid.
     *
     * \throws std::out_of_range The grid does not contain the requested tile; Either the column or
     * row index to high.
     *
     * \returns The event object of the submitted kernel.
     */
    template <typename in_pipe>
    sycl::event submit_read(sycl::queue &queue, uindex_t tile_c, uindex_t tile_r, Cell halo_value) {
        if (tile_c >= get_tile_range().c || tile_r >= get_tile_range().r) {
            throw std::out_of_range("Tile index out of range!");
        }

        constexpr uindex_t column_bits = 1 + std::bit_width(tile_width + halo_radius);
        constexpr uindex_t row_bits = 1 + std::bit_width(tile_height + halo_radius);
        using index_c_t = ac_int<column_bits, true>;
        using index_r_t = ac_int<row_bits, true>;

        return queue.submit([&](sycl::handler &cgh) {
            sycl::accessor grid_ac{grid_buffer, cgh, sycl::read_only};
            index_t grid_width = this->get_grid_width();
            index_t grid_height = this->get_grid_height();

            cgh.single_task([=]() {
                index_t c_offset = tile_c * tile_width;
                index_c_t start_tile_c = -halo_radius;
                index_c_t end_tile_c =
                    index_c_t(std::min(grid_width - tile_c * tile_width, tile_width)) + halo_radius;

                index_t r_offset = tile_r * tile_height;
                index_r_t start_tile_r = -halo_radius;
                index_r_t end_tile_r =
                    index_r_t(std::min(grid_height - tile_r * tile_height, tile_height)) +
                    halo_radius;

                [[intel::loop_coalesce(2)]] for (index_c_t tile_c = start_tile_c;
                                                 tile_c < end_tile_c; tile_c++) {
                    for (index_r_t tile_r = start_tile_r; tile_r < end_tile_r; tile_r++) {
                        index_t c = c_offset + tile_c.to_long();
                        index_t r = r_offset + tile_r.to_long();

                        Cell value;
                        if (c >= 0 && r >= 0 && c < grid_width && r < grid_height) {
                            value = grid_ac[c][r];
                        } else {
                            value = halo_value;
                        }
                        in_pipe::write(value);
                    }
                }
            });
        });
    }

    /**
     * \brief Submit a kernel that receives cells from the pipe and writes them to the grid.
     *
     * The kernel expects that one tile worth of cells can be read from the pipe. Also, it expects
     * that the cells are sent in column-major order, meaning that the last index (which denotes the
     * row) will change the quickest. The method returns the event of the launched kernel
     * immediately.
     *
     * The tile column and row indices denote the indices of the *tile*, not the cells. The starting
     * column index will be `tile_c * tile_width` and the end index will be `(tile_c + 1) *
     * tile_width`. The start and end indices for the rows are analogous.
     *
     * This method is explicitly part of the user-facing API: You are allowed and encouraged to use
     * this method to feed custom kernels.
     *
     * \tparam out_pipe The pipe the data is received from.
     *
     * \param queue The queue to submit the kernel to.
     *
     * \param tile_c The column index of the tile to read.
     *
     * \param tile_r The row index of the tile to read.
     *
     * \throws std::out_of_range The grid does not contain the requested tile; Either the column or
     * row index to high.
     *
     * \returns The event object of the submitted kernel.
     */
    template <typename out_pipe>
    sycl::event submit_write(sycl::queue queue, uindex_t tile_c, uindex_t tile_r) {
        if (tile_c >= get_tile_range().c || tile_r >= get_tile_range().r) {
            throw std::out_of_range("Tile index out of range!");
        }

        constexpr uindex_t column_bits = std::bit_width(tile_width);
        constexpr uindex_t row_bits = std::bit_width(tile_height);
        using uindex_c_t = ac_int<column_bits, false>;
        using uindex_r_t = ac_int<row_bits, false>;

        return queue.submit([&](sycl::handler &cgh) {
            sycl::accessor grid_ac{grid_buffer, cgh, sycl::read_write};
            uindex_t grid_width = this->get_grid_width();
            uindex_t grid_height = this->get_grid_height();

            cgh.single_task([=]() {
                uindex_t c_offset = tile_c * tile_width;
                uindex_c_t end_tile_c =
                    uindex_c_t(std::min(grid_width - tile_c * tile_width, tile_width));

                uindex_t r_offset = tile_r * tile_height;
                uindex_r_t end_tile_r =
                    uindex_r_t(std::min(grid_height - tile_r * tile_height, tile_height));

                [[intel::loop_coalesce(2)]] for (uindex_c_t tile_c = 0; tile_c < end_tile_c;
                                                 tile_c++) {
                    for (uindex_r_t tile_r = 0; tile_r < end_tile_r; tile_r++) {
                        grid_ac[c_offset + tile_c.to_long()][r_offset + tile_r.to_long()] =
                            out_pipe::read();
                    }
                }
            });
        });
    }

  private:
    sycl::buffer<Cell, 2> grid_buffer;
};

} // namespace tiling
} // namespace stencil
