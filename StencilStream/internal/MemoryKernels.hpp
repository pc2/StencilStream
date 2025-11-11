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
#include "Helpers.hpp"
#include <bit>
#include <exception>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/sycl.hpp>

namespace stencil {
namespace internal {

template <typename T, sycl::access::mode access_mode, std::size_t max_buffer_height,
          std::size_t max_buffer_width>
class CompleteBufferIOKernel {
  public:
    using Accessor = sycl::accessor<T, 2, access_mode, sycl::access::target::device>;
    using uindex_r_t = ac_int<std::bit_width(max_buffer_height), false>;
    using uindex_c_t = ac_int<std::bit_width(max_buffer_width), false>;

    CompleteBufferIOKernel(sycl::buffer<T, 2> buffer, sycl::handler &cgh)
        : accessor(buffer, cgh), buffer_height(buffer.get_range()[0]),
          buffer_width(buffer.get_range()[1]) {
        if (buffer.get_range()[0] > max_buffer_height || buffer.get_range()[1] > max_buffer_width) {
            throw std::out_of_range("The given buffer is too big for the IO kernel.");
        }
    }

    CompleteBufferIOKernel(Accessor accessor)
        : accessor(accessor), buffer_height(accessor.get_range()[0]),
          buffer_width(accessor.get_range()[1]) {
        if (accessor.get_range()[0] > max_buffer_height ||
            accessor.get_range()[1] > max_buffer_width) {
            throw std::out_of_range("The given buffer is too big for the IO kernel.");
        }
    }

  protected:
    Accessor accessor;
    uindex_r_t buffer_height;
    uindex_c_t buffer_width;
};

template <typename T, typename out_pipe,
          std::size_t max_buffer_height = std::numeric_limits<std::size_t>::max() - 1,
          std::size_t max_buffer_width = std::numeric_limits<std::size_t>::max() - 1>
class CompleteBufferReadKernel
    : public CompleteBufferIOKernel<T, sycl::access::mode::read, max_buffer_height,
                                    max_buffer_width> {
  public:
    using Parent =
        CompleteBufferIOKernel<T, sycl::access::mode::read, max_buffer_height, max_buffer_width>;
    using Accessor = Parent::Accessor;
    using uindex_r_t = Parent::uindex_r_t;
    using uindex_c_t = Parent::uindex_c_t;

    CompleteBufferReadKernel(sycl::buffer<T, 2> buffer, sycl::handler &cgh) : Parent(buffer, cgh) {}
    CompleteBufferReadKernel(Accessor accessor) : Parent(accessor) {}

    void operator()() const {
        auto accessor = this->accessor;
        uindex_r_t buffer_height = this->buffer_height;
        uindex_c_t buffer_width = this->buffer_width;

        [[intel::loop_coalesce(2)]]
        for (uindex_r_t r = 0; r < buffer_height; r++) {
            for (uindex_c_t c = 0; c < buffer_width; c++) {
                out_pipe::write(accessor[r][c]);
            }
        }
    }
};

template <typename T, typename in_pipe,
          std::size_t max_buffer_height = std::numeric_limits<std::size_t>::max() - 1,
          std::size_t max_buffer_width = std::numeric_limits<std::size_t>::max() - 1>
class CompleteBufferWriteKernel
    : public CompleteBufferIOKernel<T, sycl::access::mode::write, max_buffer_height,
                                    max_buffer_width> {
  public:
    using Parent =
        CompleteBufferIOKernel<T, sycl::access::mode::write, max_buffer_height, max_buffer_width>;
    using Accessor = Parent::Accessor;
    using uindex_r_t = Parent::uindex_r_t;
    using uindex_c_t = Parent::uindex_c_t;

    CompleteBufferWriteKernel(sycl::buffer<T, 2> buffer, sycl::handler &cgh)
        : Parent(buffer, cgh) {}
    CompleteBufferWriteKernel(Accessor accessor) : Parent(accessor) {}

    void operator()() const {
        auto accessor = this->accessor;
        uindex_r_t buffer_height = this->buffer_height;
        uindex_c_t buffer_width = this->buffer_width;

        [[intel::loop_coalesce(2)]]
        for (uindex_r_t r = 0; r < buffer_height; r++) {
            for (uindex_c_t c = 0; c < buffer_width; c++) {
                accessor[r][c] = in_pipe::read();
            }
        }
    }
};

template <typename T, sycl::access::mode access_mode, std::size_t max_tile_height,
          std::size_t max_tile_width>
class PartialBufferIOKernel {
  public:
    using Accessor = sycl::accessor<T, 2, access_mode, sycl::access::target::device>;
    using uindex_local_r_t = ac_int<std::bit_width(max_tile_height), false>;
    using uindex_local_c_t = ac_int<std::bit_width(max_tile_width), false>;

    PartialBufferIOKernel(Accessor accessor, sycl::id<2> offset, sycl::range<2> range)
        : accessor(accessor), row_offset(offset[0]), column_offset(offset[1]),
          tile_height(range[0]), tile_width(range[1]) {
        if (range[0] > max_tile_height || range[1] > max_tile_width) {
            throw std::out_of_range("The given tile is too big for the IO kernel.");
        }
        if (offset[0] + range[0] > accessor.get_range()[0] ||
            offset[1] + range[1] > accessor.get_range()[1]) {
            throw std::out_of_range("Requested tile out of buffer range.");
        }
    }

    PartialBufferIOKernel(sycl::buffer<T, 2> buffer, sycl::handler &cgh, sycl::id<2> offset,
                          sycl::range<2> range)
        : PartialBufferIOKernel(Accessor(buffer, cgh), offset, range) {}

  protected:
    Accessor accessor;
    std::size_t row_offset, column_offset;
    uindex_local_r_t tile_height;
    uindex_local_c_t tile_width;
};

template <typename T, typename out_pipe, std::size_t max_tile_height, std::size_t max_tile_width>
class PartialBufferReadKernel
    : public PartialBufferIOKernel<T, sycl::access::mode::read, max_tile_height, max_tile_width> {
  public:
    using Parent =
        PartialBufferIOKernel<T, sycl::access::mode::read, max_tile_height, max_tile_width>;
    using Accessor = Parent::Accessor;
    using uindex_local_r_t = Parent::uindex_local_r_t;
    using uindex_local_c_t = Parent::uindex_local_c_t;

    PartialBufferReadKernel(Accessor accessor, sycl::id<2> offset, sycl::range<2> range)
        : Parent(accessor, offset, range) {}
    PartialBufferReadKernel(sycl::buffer<T, 2> buffer, sycl::handler &cgh, sycl::id<2> offset,
                            sycl::range<2> range)
        : Parent(buffer, cgh, offset, range) {}

    void operator()() const {
        [[intel::loop_coalesce(2)]] for (uindex_local_r_t local_r = 0; local_r < this->tile_height;
                                         local_r++) {
            for (uindex_local_c_t local_c = 0; local_c < this->tile_width; local_c++) {
                std::size_t r = this->row_offset + std::size_t(local_r);
                std::size_t c = this->column_offset + std::size_t(local_c);
                out_pipe::write(this->accessor[r][c]);
            }
        }
    }
};

template <typename T, typename in_pipe, std::size_t max_tile_height, std::size_t max_tile_width>
class PartialBufferWriteKernel
    : public PartialBufferIOKernel<T, sycl::access::mode::write, max_tile_height, max_tile_width> {
  public:
    using Parent =
        PartialBufferIOKernel<T, sycl::access::mode::write, max_tile_height, max_tile_width>;
    using Accessor = Parent::Accessor;
    using uindex_local_r_t = Parent::uindex_local_r_t;
    using uindex_local_c_t = Parent::uindex_local_c_t;

    PartialBufferWriteKernel(Accessor accessor, sycl::id<2> offset, sycl::range<2> range)
        : Parent(accessor, offset, range) {}
    PartialBufferWriteKernel(sycl::buffer<T, 2> buffer, sycl::handler &cgh, sycl::id<2> offset,
                             sycl::range<2> range)
        : Parent(buffer, cgh, offset, range) {}

    void operator()() const {
        [[intel::loop_coalesce(2)]] for (uindex_local_r_t local_r = 0; local_r < this->tile_height;
                                         local_r++) {
            for (uindex_local_c_t local_c = 0; local_c < this->tile_width; local_c++) {
                std::size_t r = this->row_offset + std::size_t(local_r);
                std::size_t c = this->column_offset + std::size_t(local_c);
                this->accessor[r][c] = in_pipe::read();
            }
        }
    }
};

} // namespace internal
} // namespace stencil
