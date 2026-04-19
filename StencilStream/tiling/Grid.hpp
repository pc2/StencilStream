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
#include "../AccessorSubscript.hpp"
#include "../Concepts.hpp"
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
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
 * \tparam Cell The cell type to store.
 *
 * \tparam spatial_parallelism The number of cells to load and store in parallel. Has to match the
 * `spatial_parallelism` parameter for a \ref StencilUpdate instance.
 */
template <typename Cell, std::size_t spatial_parallelism> class Grid {
  public:
    /**
     * \brief The number of dimensions of the grid.
     *
     * May be changed in the future when other dimensions are supported.
     */
    static constexpr std::size_t dimensions = 2;

    /**
     * \brief The vectorized cell type used for internal storage.
     *
     * Groups `spatial_parallelism` cells into a single padded value for memory-aligned bulk
     * access. The padding ensures that the memory word boundary is respected when loading or
     * storing multiple cells at once.
     */
    using CellVector = stencil::internal::Padded<std::array<Cell, spatial_parallelism>>;

    /**
     * \brief Create a new, uninitialized grid with the given dimensions.
     *
     * \param grid_height The height, or number of rows, of the new grid.
     *
     * \param grid_width The width, or number of columns, of the new grid.
     */
    Grid(std::size_t grid_height, std::size_t grid_width)
        : grid_buffer(sycl::range<2>(
              grid_height, stencil::internal::int_ceil_div(grid_width, spatial_parallelism))),
          grid_range(grid_height, grid_width) {}

    /**
     * \brief Create a new, uninitialized grid with the given dimensions.
     *
     * \param range The range of the new grid. The first index will be the height and the second
     * index will be the width of the grid.
     */
    Grid(sycl::range<2> range) : Grid(range[0], range[1]) {}

    /**
     * \brief Create a new grid with the same size and contents as the given SYCL buffer.
     *
     * The contents of the buffer will be copied to the grid by the host. The SYCL buffer can later
     * be used elsewhere.
     *
     * \param input_buffer The buffer with the contents of the new grid.
     */
    Grid(sycl::buffer<Cell, 2> input_buffer)
        : Grid(input_buffer.get_range()[0], input_buffer.get_range()[1]) {
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
    Grid(Grid const &other_grid)
        : grid_buffer(other_grid.grid_buffer), grid_range(other_grid.grid_range) {}

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
        static constexpr std::size_t dimensions = Grid::dimensions;

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
            return accessor[id[0]][id[1] / spatial_parallelism].value[id[1] % spatial_parallelism];
        }

        /**
         * \brief Access a cell of the grid.
         *
         * \param id The index of the accessed cell. The first index is the row index, the second
         * one is the column index.
         *
         * \returns A reference to the indexed cell.
         */
        Cell &operator[](sycl::id<2> id)
            requires(access_mode != sycl::access::mode::read)
        {
            return accessor[id[0]][id[1] / spatial_parallelism].value[id[1] % spatial_parallelism];
        }

      private:
        sycl::host_accessor<CellVector, 2, access_mode> accessor;
    };

    /**
     * \brief Copy the contents of the SYCL buffer into the grid.
     *
     * The SYCL buffer will be accessed read-only one the host; It may be used elsewhere too. The
     * buffer however has to have the same size as the grid, otherwise a std::range_error is
     * thrown.
     *
     * \param input_buffer The buffer to copy the data from.
     * \throws std::range_error The size of the buffer does not match the grid.
     */
    void copy_from_buffer(sycl::buffer<Cell, 2> input_buffer) {
        if (input_buffer.get_range() != grid_range) {
            throw std::out_of_range("The target buffer has not the same size as the grid");
        }

        sycl::host_accessor grid_ac{grid_buffer, sycl::write_only};
        sycl::host_accessor input_ac{input_buffer, sycl::read_only};
        for (std::size_t r = 0; r < get_grid_height(); r++) {
            for (std::size_t c = 0; c < get_grid_width(); c++) {
                grid_ac[r][c / spatial_parallelism].value[c % spatial_parallelism] = input_ac[r][c];
            }
        }
    }

    /**
     * \brief Copy the contents of the grid into the SYCL buffer.
     *
     * The contents of the SYCL buffer will be overwritten on the host. The buffer also has to have
     * the same size as the grid, otherwise a std::range_error is thrown.
     *
     * \param output_buffer The buffer to copy the data to.
     * \throws std::range_error The size of the buffer does not match the grid.
     */
    void copy_to_buffer(sycl::buffer<Cell, 2> output_buffer) {
        if (output_buffer.get_range() != grid_range) {
            throw std::out_of_range("The target buffer has not the same size as the grid");
        }

        sycl::host_accessor grid_ac{grid_buffer, sycl::read_only};
        sycl::host_accessor output_ac{output_buffer, sycl::write_only};
        for (std::size_t r = 0; r < get_grid_height(); r++) {
            for (std::size_t c = 0; c < get_grid_width(); c++) {
                output_ac[r][c] =
                    grid_ac[r][c / spatial_parallelism].value[c % spatial_parallelism];
            }
        }
    }

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

    /**
     * \brief Return the width of the grid, optionally in vectorized units.
     *
     * \param vectorized If true, returns the number of \ref CellVector columns (i.e., the grid
     * width divided by `spatial_parallelism`, rounded up). If false, returns the width in
     * individual cells, identical to `get_grid_width()`.
     */
    std::size_t get_grid_width(bool vectorized) const {
        using namespace stencil::internal;
        return vectorized ? int_ceil_div(grid_range[1], spatial_parallelism) : grid_range[1];
    }

    /**
     * \brief Return the dimensions of the grid as a two-element range.
     *
     * The first element is the height (number of rows) and the second element is the width (number
     * of columns), both measured in individual cells.
     */
    sycl::range<2> get_grid_range() const { return grid_range; }

    /**
     * \brief Return the dimensions of the grid, optionally in vectorized units.
     *
     * \param vectorized If true, the column dimension is expressed in \ref CellVector units
     * (i.e., divided by `spatial_parallelism`, rounded up). If false, individual cell units are
     * used and the result is identical to the no-argument overload.
     */
    sycl::range<2> get_grid_range(bool vectorized) const {
        if (vectorized) {
            return sycl::range<2>(grid_range[0], get_grid_width(true));
        } else {
            return grid_range;
        }
    }

    /**
     * \brief Return the number of tiles in each dimension when the grid is partitioned into tiles.
     *
     * The grid is logically divided into a rectangular arrangement of tiles, each at most
     * `max_tile_range` in size. Boundary tiles may be smaller. This method returns the number of
     * tile columns and tile rows.
     *
     * \param max_tile_range The maximum dimensions of a single tile, in individual cells. The
     * column dimension must be a multiple of `spatial_parallelism`.
     * \return A two-element range whose first element is the number of tile rows and whose second
     * element is the number of tile columns.
     * \throws std::invalid_argument The tile width is not a multiple of `spatial_parallelism`.
     */
    sycl::range<2> get_tile_id_range(sycl::range<2> max_tile_range) const {
        using namespace stencil::internal;
        if (max_tile_range[1] % spatial_parallelism != 0) {
            throw std::invalid_argument(
                "Tile widths must be a multiple of the spatial parallelism");
        }
        return sycl::range<2>(int_ceil_div(get_grid_height(), max_tile_range[0]),
                              int_ceil_div(get_grid_width(), max_tile_range[1]));
    }

    /**
     * \brief Return the grid offset of the top-left corner of a tile.
     *
     * \param tile_id The row/column index of the tile in the tile grid.
     * \param max_tile_range The maximum dimensions of a single tile, in individual cells.
     * \param vectorized If true, the returned column offset is expressed in \ref CellVector units.
     * If false, it is expressed in individual cells.
     * \return The offset (row, column) of the tile's top-left corner within the grid.
     * \throws std::out_of_range `tile_id` is outside the tile grid.
     */
    sycl::id<2> get_tile_offset(sycl::id<2> tile_id, sycl::range<2> max_tile_range,
                                bool vectorized) const {
        sycl::range<2> tile_id_range = get_tile_id_range(max_tile_range);
        if (tile_id[0] >= tile_id_range[0] || tile_id[1] >= tile_id_range[1]) {
            throw std::out_of_range("Tile ID out of range");
        }
        std::size_t row_offset = tile_id[0] * max_tile_range[0];
        std::size_t column_offset = tile_id[1] * max_tile_range[1];
        if (vectorized) {
            column_offset /= spatial_parallelism;
        }
        return sycl::id<2>(row_offset, column_offset);
    }

    /**
     * \brief Return the grid offset of the top-left corner of a tile including its halo region.
     *
     * The halo region extends `halo_range` cells beyond each edge of the tile, providing the
     * stencil neighborhood for cells at tile boundaries. For the first tile row/column the haloed
     * offset may be negative (i.e., before the grid boundary).
     *
     * \param tile_id The row/column index of the tile in the tile grid.
     * \param max_tile_range The maximum dimensions of a single tile, in individual cells.
     * \param halo_range The halo size in each dimension, in individual cells. The column dimension
     * must be a multiple of `spatial_parallelism`.
     * \param vectorized If true, the returned column offset is expressed in \ref CellVector units.
     * If false, it is expressed in individual cells.
     * \param only_valid_cells If true, clamp the returned offset so it is never negative (i.e.,
     * never before the grid origin).
     * \return A two-element array {row_offset, column_offset} of the haloed tile's top-left corner.
     * \throws std::invalid_argument The halo width is not a multiple of `spatial_parallelism`.
     */
    std::array<std::ptrdiff_t, 2> get_haloed_tile_offset(sycl::id<2> tile_id,
                                                         sycl::range<2> max_tile_range,
                                                         sycl::range<2> halo_range, bool vectorized,
                                                         bool only_valid_cells) const {
        if (halo_range[1] % spatial_parallelism != 0) {
            throw std::invalid_argument(
                "Halo widths must be a multiple of the spatial parallelism");
        }
        sycl::id<2> tile_offset = get_tile_offset(tile_id, max_tile_range, vectorized);
        std::ptrdiff_t haloed_row_offset = tile_offset[0] - halo_range[0];
        std::ptrdiff_t haloed_column_offset =
            tile_offset[1] - (vectorized ? halo_range[1] / spatial_parallelism : halo_range[1]);
        if (only_valid_cells) {
            haloed_row_offset = std::max<std::ptrdiff_t>(haloed_row_offset, 0);
            haloed_column_offset = std::max<std::ptrdiff_t>(haloed_column_offset, 0);
        }
        return {haloed_row_offset, haloed_column_offset};
    }

    /**
     * \brief Return the core dimensions of a tile.
     *
     * Interior tiles have the dimensions given by `max_tile_range`. South and east boundary tiles
     * that do not fill a full `max_tile_range` region are smaller.
     *
     * \param tile_id The row/column index of the tile in the tile grid.
     * \param max_tile_range The maximum dimensions of a single tile, in individual cells.
     * \param vectorized If true, the returned column dimension is expressed in \ref CellVector
     * units. If false, it is expressed in individual cells.
     * \return The actual range (height, width) of the tile.
     * \throws std::out_of_range `tile_id` is outside the tile grid.
     */
    sycl::range<2> get_tile_range(sycl::id<2> tile_id, sycl::range<2> max_tile_range,
                                  bool vectorized) const {
        using namespace stencil::internal;
        sycl::range<2> tile_id_range = get_tile_id_range(max_tile_range);
        if (tile_id[0] >= tile_id_range[0] || tile_id[1] >= tile_id_range[1]) {
            throw std::out_of_range("Tile ID out of range");
        }
        std::size_t tile_height =
            std::min(grid_range[0] - tile_id[0] * max_tile_range[0], max_tile_range[0]);
        std::size_t tile_width =
            std::min(grid_range[1] - tile_id[1] * max_tile_range[1], max_tile_range[1]);
        if (vectorized) {
            tile_width = int_ceil_div(tile_width, spatial_parallelism);
        }
        return sycl::range<2>(tile_height, tile_width);
    }

    /**
     * \brief Return the dimensions of a tile including its halo region.
     *
     * The returned range covers the tile core itself plus `halo_range` extra cells on each side.
     * At grid boundaries, the halo may extend beyond the grid; `only_valid_cells` controls
     * whether those out-of-bounds portions are clipped.
     *
     * \param tile_id The row/column index of the tile in the tile grid.
     * \param max_tile_range The maximum dimensions of a single tile, in individual cells.
     * \param halo_range The halo size in each dimension, in individual cells. The column dimension
     * must be a multiple of `spatial_parallelism`.
     * \param vectorized If true, the returned column dimension is expressed in \ref CellVector
     * units. If false, it is expressed in individual cells.
     * \param only_valid_cells If true, clip the haloed range so it does not extend beyond the
     * grid boundaries.
     * \return The range (height, width) of the haloed tile.
     * \throws std::invalid_argument The halo width is not a multiple of `spatial_parallelism`.
     */
    sycl::range<2> get_haloed_tile_range(sycl::id<2> tile_id, sycl::range<2> max_tile_range,
                                         sycl::range<2> halo_range, bool vectorized,
                                         bool only_valid_cells) const {
        if (halo_range[1] % spatial_parallelism != 0) {
            throw std::invalid_argument(
                "Halo widths must be a multiple of the spatial parallelism");
        }
        sycl::range<2> haloed_tile_range = get_tile_range(tile_id, max_tile_range, vectorized);
        haloed_tile_range[0] += 2 * halo_range[0];
        haloed_tile_range[1] +=
            2 * (vectorized ? halo_range[1] / spatial_parallelism : halo_range[1]);

        if (only_valid_cells) {
            std::array<std::ptrdiff_t, 2> haloed_tile_offset =
                get_haloed_tile_offset(tile_id, max_tile_range, halo_range, vectorized, false);
            sycl::range<2> grid_range = get_grid_range(vectorized);
            sycl::id<2> lower_right_corner(
                std::min(haloed_tile_offset[0] + haloed_tile_range[0], grid_range[0]),
                std::min(haloed_tile_offset[1] + haloed_tile_range[1], grid_range[1]));
            std::array<std::ptrdiff_t, 2> upper_left_corner =
                get_haloed_tile_offset(tile_id, max_tile_range, halo_range, vectorized, true);
            return sycl::range<2>(lower_right_corner[0] - upper_left_corner[0],
                                  lower_right_corner[1] - upper_left_corner[1]);
        } else {
            return haloed_tile_range;
        }
    }

    /**
     * \brief Return the underlying vectorized grid buffer.
     *
     * Returns the internal SYCL buffer whose cells are stored in \ref CellVector units.
     * This is used by the \ref StencilUpdate backend to submit read and write kernels directly
     * against the vectorized storage layout.
     */
    sycl::buffer<CellVector, 2> get_internal() const { return grid_buffer; }

  private:
    sycl::buffer<CellVector, 2> grid_buffer;
    sycl::range<2> grid_range;
};

} // namespace tiling
} // namespace stencil
