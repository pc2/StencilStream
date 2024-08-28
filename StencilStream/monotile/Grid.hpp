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
 * \tparam word_size The word size of the memory system, in bytes. This is used to optimize the
 * kernels submitted by \ref submit_read and \ref submit_write.
 */
template <class Cell, uindex_t word_size = 64> class Grid {
  private:
    static constexpr uindex_t word_length =
        std::lcm(sizeof(Padded<Cell>), word_size) / sizeof(Padded<Cell>);
    using IOWord = std::array<Padded<Cell>, word_length>;

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
        : tile_buffer(sycl::range<1>(n_cells_to_n_words(grid_width * grid_height, word_length))),
          grid_width(grid_width), grid_height(grid_height) {}

    /**
     * \brief Create a new, uninitialized grid with the given dimensions.
     *
     * \param range The range of the new grid. The first index will be the width and the second
     * index will be the height of the grid.
     */
    Grid(sycl::range<2> range)
        : tile_buffer(sycl::range<1>(n_cells_to_n_words(range[0] * range[1], word_length))),
          grid_width(range[0]), grid_height(range[1]) {}

    /**
     * \brief Create a new grid with the same size and contents as the given SYCL buffer.
     *
     * The contents of the buffer will be copied to the grid by the host. The SYCL buffer can later
     * be used elsewhere.
     *
     * \param buffer The buffer with the contents of the new grid.
     */
    Grid(sycl::buffer<Cell, 2> buffer)
        : tile_buffer(1), grid_width(buffer.get_range()[0]), grid_height(buffer.get_range()[1]) {
        tile_buffer = sycl::range<1>(n_cells_to_n_words(grid_width * grid_height, word_length));
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
        : tile_buffer(other_grid.tile_buffer), grid_width(other_grid.grid_width),
          grid_height(other_grid.grid_height) {}

    /**
     * \brief Create an new, uninitialized grid with the same size as the current one.
     */
    Grid make_similar() const { return Grid(grid_width, grid_height); }

    /**
     * \brief Return the width, or number of columns, of the grid.
     */
    uindex_t get_grid_width() const { return grid_width; }

    /**
     * \brief Return the height, or number of rows, of the grid.
     */
    uindex_t get_grid_height() const { return grid_height; }

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
        using accessor_t = sycl::host_accessor<IOWord, 1, access_mode>;

      public:
        /**
         * \brief The number of dimensions of the underlying grid.
         */
        static constexpr uindex_t dimensions = Grid::dimensions;

        /**
         * \brief Create a new accessor to the given grid.
         */
        GridAccessor(Grid &grid)
            : ac(grid.tile_buffer), grid_width(grid.get_grid_width()),
              grid_height(grid.get_grid_height()) {}

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
            uindex_t word_i = (id[0] * grid_height + id[1]) / word_length;
            uindex_t cell_i = (id[0] * grid_height + id[1]) % word_length;
            return ac[word_i][cell_i].value;
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
            uindex_t word_i = (id[0] * grid_height + id[1]) / word_length;
            uindex_t cell_i = (id[0] * grid_height + id[1]) % word_length;
            return ac[word_i][cell_i].value;
        }

      private:
        accessor_t ac;
        uindex_t grid_width, grid_height;
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
        uindex_t width = this->get_grid_width();
        uindex_t height = this->get_grid_height();

        if (input_buffer.get_range() != sycl::range<2>(width, height)) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        sycl::host_accessor in_ac(input_buffer, sycl::read_only);
        GridAccessor<sycl::access::mode::read_write> tile_ac(*this);
        for (uindex_t c = 0; c < width; c++) {
            for (uindex_t r = 0; r < height; r++) {
                tile_ac[c][r] = in_ac[c][r];
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
        uindex_t width = this->get_grid_width();
        uindex_t height = this->get_grid_height();

        if (output_buffer.get_range() != sycl::range<2>(width, height)) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        GridAccessor<sycl::access::mode::read> in_ac(*this);
        sycl::host_accessor out_ac(output_buffer, sycl::write_only);
        for (uindex_t c = 0; c < width; c++) {
            for (uindex_t r = 0; r < height; r++) {
                out_ac[c][r] = in_ac[c][r];
            }
        }
    }

    /**
     * \brief Submit a kernel that sends the contents of the grid into a pipe.
     *
     * The entirety of the grid will be send into the pipe in column-major order, meaning that the
     * last index (which denotes the row) will change the quickest. The method returns the event of
     * the launched kernel immediately.
     *
     * This method is explicitly part of the user-facing API: You are allowed and encouraged to use
     * this method to feed custom kernels.
     *
     * \tparam in_pipe The pipe the data is sent into.
     *
     * \param queue The queue to submit the kernel to.
     *
     * \returns The event object of the submitted kernel.
     */
    template <typename in_pipe> sycl::event submit_read(sycl::queue queue) {
        return queue.submit([&](sycl::handler &cgh) {
            sycl::accessor ac(tile_buffer, cgh, sycl::read_only);
            uindex_t n_cells = grid_width * grid_height;

            cgh.single_task([=]() {
                IOWord cache;

                uindex_t word_i = 0;
                uindex_t cell_i = word_length;
                for (uindex_t i = 0; i < n_cells; i++) {
                    if (cell_i == word_length) {
                        cache = ac[word_i];
                        word_i++;
                        cell_i = 0;
                    }
                    in_pipe::write(cache[cell_i].value);
                    cell_i++;
                }
            });
        });
    }

    /**
     * \brief Submit a kernel that receives cells from the pipe and writes them to the grid.
     *
     * The kernel expects that the entirety of the grid can be overwritten with the cells read from
     * the pipe. Also, it expects that the cells are sent in column-major order, meaning that the
     * last index (which denotes the row) will change the quickest. The method returns the event of
     * the launched kernel immediately.
     *
     * This method is explicitly part of the user-facing API: You are allowed and encouraged to use
     * this method to feed custom kernels.
     *
     * \tparam out_pipe The pipe the data is received from.
     *
     * \param queue The queue to submit the kernel to.
     *
     * \returns The event object of the submitted kernel.
     */
    template <typename out_pipe> sycl::event submit_write(sycl::queue queue) {
        return queue.submit([&](sycl::handler &cgh) {
            sycl::accessor ac(tile_buffer, cgh, sycl::write_only);
            uindex_t n_cells = grid_width * grid_height;

            cgh.single_task([=]() {
                IOWord cache;

                uindex_t word_i = 0;
                uindex_t cell_i = 0;
                for (uindex_t i = 0; i < n_cells; i++) {
                    cache[cell_i].value = out_pipe::read();
                    cell_i++;
                    if (cell_i == word_length || i == n_cells - 1) {
                        ac[word_i] = cache;
                        cell_i = 0;
                        word_i++;
                    }
                }
            });
        });
    }

  private:
    sycl::buffer<IOWord, 1> tile_buffer;
    uindex_t grid_width, grid_height;
};
} // namespace monotile
} // namespace stencil