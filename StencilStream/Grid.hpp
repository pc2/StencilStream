/*
 * Copyright © 2020-2022 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "Concepts.hpp"
#include "Index.hpp"
#include <CL/sycl.hpp>

namespace stencil {
template <typename T>
    requires std::semiregular<T>
class Grid {
  public:
    Grid(uindex_t width, uindex_t height) : grid_range(width, height) {}
    Grid(UID grid_range) : grid_range(grid_range) {}
    Grid(cl::sycl::range<2> grid_range) : grid_range(grid_range[0], grid_range[1]) {}
    Grid(cl::sycl::buffer<T, 2> input_buffer)
        : grid_range(input_buffer.get_range()[0], input_buffer.get_range()[1]) {}

    UID get_grid_range() const { return grid_range; }

    uindex_t get_grid_width() const { return grid_range.c; }

    uindex_t get_grid_height() const { return grid_range.r; }

    virtual void copy_from_buffer(cl::sycl::buffer<T, 2> input_buffer) = 0;

    virtual void copy_to_buffer(cl::sycl::buffer<T, 2> output_buffer) = 0;

  private:
    UID grid_range;
};
} // namespace stencil