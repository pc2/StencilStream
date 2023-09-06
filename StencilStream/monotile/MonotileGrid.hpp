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
#include "../Concepts.hpp"
#include "../Padded.hpp"

namespace stencil {
namespace monotile {
template <class Cell, uindex_t tile_width, uindex_t tile_height, uindex_t word_size>
class MonotileGrid {
  public:
    static constexpr uindex_t word_length =
        std::lcm(sizeof(Padded<Cell>), word_size) / sizeof(Padded<Cell>);
    static constexpr uindex_t max_n_words =
        n_cells_to_n_words(tile_width * tile_height, word_length);

    using IOWord = std::array<Padded<Cell>, word_length>;

    static constexpr unsigned long bits_cell = std::bit_width(word_length);
    using index_cell_t = ac_int<bits_cell + 1, true>;
    using uindex_cell_t = ac_int<bits_cell, false>;

    static constexpr unsigned long bits_word = std::bit_width(max_n_words);
    using index_word_t = ac_int<bits_word + 1, true>;
    using uindex_word_t = ac_int<bits_word, false>;

    MonotileGrid(uindex_t grid_width, uindex_t grid_height)
        : tile_buffer(
              cl::sycl::range<1>(n_cells_to_n_words(grid_width * grid_height, word_length))),
          grid_width(grid_width), grid_height(grid_height) {
        if (grid_width > tile_width || grid_height > tile_height) {
            throw std::range_error("The grid is bigger than the tile. The monotile architecture "
                                   "requires that grid ranges are smaller or equal to the tile "
                                   "range");
        }
    }

    MonotileGrid(cl::sycl::buffer<Cell, 2> buffer)
        : tile_buffer(1), grid_width(buffer.get_range()[0]), grid_height(buffer.get_range()[1]) {
        if (grid_width > tile_width || grid_height > tile_height) {
            throw std::range_error("The grid is bigger than the tile. The monotile architecture "
                                   "requires that grid ranges are smaller or equal to the tile "
                                   "range");
        }
        tile_buffer = cl::sycl::range<1>(n_cells_to_n_words(grid_width * grid_height, word_length));
        copy_from_buffer(buffer);
    }

    MonotileGrid make_similar() const { return MonotileGrid(grid_width, grid_height); }

    uindex_t get_grid_width() const { return grid_width; }

    uindex_t get_grid_height() const { return grid_height; }

    void copy_from_buffer(cl::sycl::buffer<Cell, 2> input_buffer) {
        uindex_t width = this->get_grid_width();
        uindex_t height = this->get_grid_height();

        if (input_buffer.get_range() != cl::sycl::range<2>(width, height)) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        auto in_ac = input_buffer.template get_access<cl::sycl::access::mode::read>();
        auto tile_ac = tile_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        for (uindex_t c = 0; c < width; c++) {
            for (uindex_t r = 0; r < height; r++) {
                uindex_t word_i = (c * height + r) / word_length;
                uindex_t cell_i = (c * height + r) % word_length;
                tile_ac[word_i][cell_i].value = in_ac[c][r];
            }
        }
    }

    void copy_to_buffer(cl::sycl::buffer<Cell, 2> output_buffer) {
        uindex_t width = this->get_grid_width();
        uindex_t height = this->get_grid_height();

        if (output_buffer.get_range() != cl::sycl::range<2>(width, height)) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }

        auto in_ac = tile_buffer.template get_access<cl::sycl::access::mode::read>();
        auto out_ac = output_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        for (uindex_t c = 0; c < output_buffer.get_range()[0]; c++) {
            for (uindex_t r = 0; r < output_buffer.get_range()[1]; r++) {
                uindex_t word_i = (c * height + r) / word_length;
                uindex_t cell_i = (c * height + r) % word_length;
                out_ac[c][r] = in_ac[word_i][cell_i].value;
            }
        }
    }

    template <typename in_pipe> void submit_read(cl::sycl::queue queue) {
        queue.submit([&](cl::sycl::handler &cgh) {
            auto ac = tile_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
            uindex_t n_cells = this->get_grid_width() * this->get_grid_height();

            cgh.single_task([=]() {
                [[intel::fpga_memory]] IOWord cache;

                uindex_word_t word_i = 0;
                uindex_cell_t cell_i = word_length;
                for (uindex_t i = 0; i < n_cells; i++) {
                    if (cell_i == uindex_cell_t(word_length)) {
                        cache = ac[word_i.to_uint64()];
                        word_i++;
                        cell_i = 0;
                    }
                    in_pipe::write(cache[cell_i].value);
                    cell_i++;
                }
            });
        });
    }

    template <typename out_pipe> void submit_write(cl::sycl::queue queue) {
        queue.submit([&](cl::sycl::handler &cgh) {
            auto ac = tile_buffer.template get_access<cl::sycl::access::mode::discard_write>(cgh);
            uindex_t n_cells = this->get_grid_width() * this->get_grid_height();

            cgh.single_task([=]() {
                [[intel::fpga_memory]] IOWord cache;

                uindex_word_t word_i = 0;
                uindex_cell_t cell_i = 0;
                for (uindex_t i = 0; i < n_cells; i++) {
                    cache[cell_i].value = out_pipe::read();
                    cell_i++;
                    if (cell_i == uindex_cell_t(word_length)) {
                        ac[word_i.to_uint64()] = cache;
                        cell_i = 0;
                        word_i++;
                    }
                }

                if (cell_i != 0) {
                    ac[word_i.to_uint64()] = cache;
                }
            });
        });
    }

  private:
    cl::sycl::buffer<IOWord, 1> tile_buffer;
    uindex_t grid_width, grid_height;
};
} // namespace monotile
} // namespace stencil