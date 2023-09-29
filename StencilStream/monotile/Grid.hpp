/*
 * Copyright © 2020-2023 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "../Padded.hpp"

namespace stencil {
namespace monotile {

template <class Cell, uindex_t word_size = 64> class Grid {
  public:
    static constexpr uindex_t dimensions = 2;
    static constexpr uindex_t word_length =
        std::lcm(sizeof(Padded<Cell>), word_size) / sizeof(Padded<Cell>);

    using IOWord = std::array<Padded<Cell>, word_length>;

    Grid(uindex_t grid_width, uindex_t grid_height)
        : tile_buffer(sycl::range<1>(n_cells_to_n_words(grid_width * grid_height, word_length))),
          grid_width(grid_width), grid_height(grid_height) {}

    Grid(sycl::buffer<Cell, 2> buffer)
        : tile_buffer(1), grid_width(buffer.get_range()[0]), grid_height(buffer.get_range()[1]) {
        tile_buffer = sycl::range<1>(n_cells_to_n_words(grid_width * grid_height, word_length));
        copy_from_buffer(buffer);
    }

    Grid make_similar() const { return Grid(grid_width, grid_height); }

    uindex_t get_grid_width() const { return grid_width; }

    uindex_t get_grid_height() const { return grid_height; }

    template <sycl::access::mode access_mode = sycl::access::mode::read_write> class GridAccessor {
      public:
        using accessor_t = sycl::host_accessor<IOWord, 1, access_mode>;
        static constexpr uindex_t dimensions = Grid::dimensions;

        GridAccessor(Grid &grid)
            : ac(grid.tile_buffer), grid_width(grid.get_grid_width()),
              grid_height(grid.get_grid_height()) {}

        using BaseSubscript = AccessorSubscript<Cell, GridAccessor, access_mode>;
        BaseSubscript operator[](uindex_t i) { return BaseSubscript(*this, i); }

        Cell const &operator[](sycl::id<2> id)
            requires(access_mode == sycl::access::mode::read)
        {
            uindex_t word_i = (id[0] * grid_height + id[1]) / word_length;
            uindex_t cell_i = (id[0] * grid_height + id[1]) % word_length;
            return ac[word_i][cell_i].value;
        }

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

    template <typename in_pipe> void submit_read(sycl::queue queue) {
        queue.submit([&](sycl::handler &cgh) {
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

    template <typename out_pipe> void submit_write(sycl::queue queue) {
        queue.submit([&](sycl::handler &cgh) {
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