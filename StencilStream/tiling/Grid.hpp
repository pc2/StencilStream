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
#include <memory>
#include <numeric>
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
 * On the device side, the data can be read or written with the help of the method templates \ref
 * submit_read and \ref submit_write. Those take a SYCL pipe as a template argument and enqueue
 * kernels that read/write the contents of the grid to/from the pipes.
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

    using CellVector = Padded<std::array<Cell, spatial_parallelism>>;

    /**
     * \brief Create a new, uninitialized grid with the given dimensions.
     *
     * \param grid_height The height, or number of rows, of the new grid.
     *
     * \param grid_width The width, or number of columns, of the new grid.
     */
    Grid(std::size_t grid_height, std::size_t grid_width)
        : grid_buffer(sycl::range<2>(grid_height, int_ceil_div(grid_width, spatial_parallelism))),
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
     * buffer however has to have the same size as the grid, otherwise a \ref std::range_error is
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
     * the same size as the grid, otherwise a \ref std::range_error is thrown.
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

    std::size_t get_vect_grid_width() const {
        return int_ceil_div(grid_range[1], spatial_parallelism);
    }

    sycl::range<2> get_grid_range() const { return grid_range; }

    sycl::range<2> get_vect_grid_range() const {
        return sycl::range<2>(grid_range[0], get_vect_grid_width());
    }

    /**
     * \brief Return the range of (central) tiles of the grid.
     *
     * This is not the range of a single tile nor is the range of the grid. It is the range of valid
     * arguments for \ref submit_read and \ref submit_write. For example, if the grid is 60 by 60
     * cells in size and a tile is 32 by 32 cells in size, the tile range would be 2 by 2 tiles.
     *
     * \return The range of tiles of the grid.
     */
    sycl::range<2> get_tile_id_range(std::size_t tile_height, std::size_t tile_width) const {
        return sycl::range<2>(int_ceil_div(get_grid_height(), tile_height),
                              int_ceil_div(get_grid_width(), tile_width));
    }

    template <std::size_t tile_height, std::size_t tile_width>
    sycl::range<2> get_tile_id_range() const {
        return get_tile_id_range(tile_height, tile_width);
    }

    /**
     * \brief Submit a kernel that sends a tile of the grid into a pipe.
     *
     * The submitted kernel will send the contents of a tile, along with the required halo, into a
     * pipe in row-major order. This means that the last index (which denotes the column) will
     * change the quickest. The method returns the event of the launched kernel immediately.
     *
     * The tile row and column indices refer to *tiles*, not the cells. The first
     * row index will be `tile_r * tile_height - halo_radius` and the end row index will be `(tile_r
     * + 1) * tile_height + halo_radius`. The start and end indices for the rows are analogous.
     *
     * This method is explicitly part of the user-facing API: You are allowed and encouraged to use
     * this method to feed custom kernels.
     *
     * \tparam in_pipe The pipe the data is sent into.
     *
     * \param queue The queue to submit the kernel to.
     *
     * \param tile_r The row index of the tile to read.
     *
     * \param tile_c The column index of the tile to read.
     *
     * \param halo_value The value to present for cells outside of the grid.
     *
     * \throws std::out_of_range The grid does not contain the requested tile; Either the column or
     * row index to high.
     *
     * \returns The event object of the submitted kernel.
     */
    template <typename in_pipe, std::size_t tile_height, std::size_t tile_width,
              std::size_t halo_height, std::size_t halo_width>
    sycl::event submit_read(sycl::queue &queue, std::size_t tile_r, std::size_t tile_c,
                            Cell halo_value) {
        static_assert(2 * halo_height < tile_height && 2 * halo_width < tile_width);
        static_assert(halo_width % spatial_parallelism == 0);
        static_assert(tile_width % spatial_parallelism == 0);

        sycl::range<2> tile_range = get_tile_id_range(tile_height, tile_width);
        std::size_t grid_height = this->get_grid_height();
        std::size_t grid_width = this->get_grid_width();

        if (tile_r >= tile_range[0] || tile_c >= tile_range[1]) {
            throw std::out_of_range("Tile index out of range!");
        }

        constexpr std::size_t row_bits = std::bit_width(tile_height + 2 * halo_height);
        constexpr std::size_t column_bits = std::bit_width(tile_width + 2 * halo_width);
        using uindex_r_t = ac_int<row_bits, false>;
        using uindex_c_t = ac_int<column_bits, false>;

        return queue.submit([&](sycl::handler &cgh) {
            sycl::accessor grid_ac{grid_buffer, cgh, sycl::read_only};

            cgh.single_task([=]() {
                std::ptrdiff_t r_offset = tile_r * tile_height - halo_height;
                uindex_r_t output_tile_height =
                    std::min(tile_height, grid_height - tile_r * tile_height);
                uindex_r_t input_tile_height = output_tile_height + 2 * halo_height;

                uindex_c_t vect_halo_width = halo_width / spatial_parallelism;
                std::size_t vect_grid_width = int_ceil_div(grid_width, spatial_parallelism);
                std::ptrdiff_t vect_c_offset =
                    (tile_c * tile_width - halo_width) / spatial_parallelism;
                uindex_c_t output_tile_width =
                    std::min(tile_width, grid_width - tile_c * tile_width);
                uindex_c_t vect_output_tile_width =
                    int_ceil_div<std::size_t>(output_tile_width, spatial_parallelism);
                uindex_c_t vect_input_tile_width = vect_output_tile_width + 2 * vect_halo_width;

                bool is_left_border_tile = tile_c == 0;
                bool is_top_border_tile = tile_r == 0;
                bool is_right_border_tile = tile_c == tile_range[1] - 1;
                bool is_bottom_border_tile = tile_r == tile_range[0] - 1;

                [[intel::loop_coalesce(2)]] for (uindex_r_t local_r = 0;
                                                 local_r < input_tile_height; local_r++) {
                    for (uindex_c_t vect_local_c = 0; vect_local_c < vect_input_tile_width;
                         vect_local_c++) {
                        std::size_t r = r_offset + std::size_t(local_r);
                        std::size_t vect_c = vect_c_offset + std::size_t(vect_local_c);

                        bool is_halo_vector = is_left_border_tile && vect_local_c < vect_halo_width;
                        is_halo_vector |= is_top_border_tile && local_r < halo_height;
                        is_halo_vector |=
                            is_right_border_tile &&
                            vect_local_c >= (vect_output_tile_width + vect_halo_width);
                        is_halo_vector |=
                            is_bottom_border_tile && local_r >= (output_tile_height + halo_height);

                        CellVector vector;
                        if (is_halo_vector) {
#pragma unroll
                            for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                                vector.value[i_cell] = halo_value;
                            }
                        } else {
                            vector = grid_ac[r][vect_c];
#pragma unroll
                            for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                                // Additional masking of halo cells if the right-most vector is
                                // incomplete.
                                if (is_right_border_tile &&
                                    vect_local_c * spatial_parallelism + i_cell >=
                                        output_tile_width + halo_width) {
                                    vector.value[i_cell] = halo_value;
                                }
                            }
                        }

                        in_pipe::write(vector);
                    }
                }
            });
        });
    }

    sycl::buffer<CellVector, 2> get_internal() const { return grid_buffer; }

  private:
    sycl::buffer<CellVector, 2> grid_buffer;
    sycl::range<2> grid_range;
};

} // namespace tiling
} // namespace stencil
