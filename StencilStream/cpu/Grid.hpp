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
#include "../Index.hpp"
#include <memory>

namespace stencil {
namespace cpu {
template <typename Cell> class Grid {
  public:
    Grid(uindex_t c, uindex_t r) : buffer(sycl::range<2>(c, r)) {}

    Grid(sycl::buffer<Cell, 2> other_buffer) : buffer(other_buffer.get_range()) {
        copy_from_buffer(other_buffer);
    }

    void copy_from_buffer(sycl::buffer<Cell, 2> other_buffer) {
        if (buffer.get_range() != other_buffer.get_range()) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }
        sycl::host_accessor buffer_ac(buffer, sycl::write_only);
        sycl::host_accessor other_ac(other_buffer, sycl::read_only);
        std::memcpy(buffer_ac.get_pointer(), other_ac.get_pointer(), buffer_ac.byte_size());
    }

    void copy_to_buffer(sycl::buffer<Cell, 2> other_buffer) {
        if (buffer.get_range() != other_buffer.get_range()) {
            throw std::range_error("The target buffer has not the same size as the grid");
        }
        sycl::host_accessor buffer_ac(buffer, sycl::read_only);
        sycl::host_accessor other_ac(other_buffer, sycl::write_only);
        std::memcpy(other_ac.get_pointer(), buffer_ac.get_pointer(), buffer_ac.byte_size());
    }

    template <sycl::access::mode access_mode = sycl::access::mode::read_write> class GridAccessor {
      public:
        GridAccessor(Grid &grid) : ac(grid.buffer) {}

        Cell get(uindex_t c, uindex_t r) const { return ac[c][r]; }

        void set(uindex_t c, uindex_t r, Cell cell) { ac[c][r] = cell; }

      private:
        sycl::host_accessor<Cell, 2, access_mode> ac;
    };

    uindex_t get_grid_width() const { return buffer.get_range()[0]; }

    uindex_t get_grid_height() const { return buffer.get_range()[1]; }

    Grid make_similar() const { return Grid(get_grid_width(), get_grid_height()); }

    sycl::buffer<Cell, 2> &get_buffer() { return buffer; }

  private:
    sycl::buffer<Cell, 2> buffer;
};
} // namespace cpu
} // namespace stencil