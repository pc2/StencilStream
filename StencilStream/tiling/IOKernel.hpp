/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "../CounterID.hpp"
#include "../Index.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/accessor.hpp>
#include <array>

namespace stencil {
namespace tiling {

/**
 * \brief Generic Input/Output kernel for use with the \ref ExecutionKernel and \ref Grid.
 *
 * This kernel provides IO services to the execution kernel by writing the contents of a input tile
 * with halo to the input pipe and writing the output of the execution kernel to an output tile. The
 * input and output code only differs by one line. Therefore, both services are provided by the same
 * class. Unlike \ref ExecutionKernel, this kernel is supposed to be constructed by a lambda
 * expression that then either executes \ref IOKernel.read or \ref IOKernel.write.
 *
 * Logically, an instance of the IO kernel receives access to a horizontal slice of the input or
 * output and processes this slice in the \ref indexingorder. Due to the \ref tiling, this slice is
 * partioned vertically into 2*n + 1 buffers, where n is the `n_halo_height_buffers` template
 * parameter. For the input buffer, n should be 2, and for the output buffer, n should be 1. All
 * buffers are expected to have a static height and dynamic width. The upper n and lower n buffers
 * are all expected be `halo_height` cells high, while the buffer in the middle is expected to have
 * be `core_height` cells high.
 *
 * \tparam T Cell value type.
 * \tparam halo_height The radius (aka width and height) of the tile halo (and therefore both
 * dimensions of a corner buffer).
 * \tparam core_height The height of the core buffer.
 * \tparam pipe The pipe to read or write to.
 * \tparam n_halo_height_buffers The number of buffers to accept, in addition to the core buffer.
 * \tparam access_mode The access mode to expect for the buffer accessor.
 * \tparam access_target The access target to expect for the buffer accessor.
 */
template <typename T, uindex_1d_t halo_height, uindex_1d_t core_height, typename pipe,
          uindex_min_t n_halo_height_buffers, cl::sycl::access::mode access_mode,
          cl::sycl::access::target access_target, uindex_t burst_buffer_length>
class IOKernel {
  public:
    /**
     * \brief The exact accessor type required by the IO kernel.
     */
    using Accessor = cl::sycl::accessor<T[burst_buffer_length], 1, access_mode, access_target>;

    /**
     * \brief The total number of buffers in a slice/column.
     */
    static constexpr uindex_min_t n_buffers = 2 * n_halo_height_buffers + 1;
    /**
     * \brief The total number of cell rows in the (logical) slice/column.
     */
    static constexpr uindex_1d_t n_rows = 2 * n_halo_height_buffers * halo_height + core_height;

    /**
     * \brief Get the height of a buffer.
     *
     * As described in the description, the first and last n buffers are `halo_height` cells high,
     * while the middle buffer is `core_height` cells high.
     */
    static constexpr uindex_1d_t get_buffer_height(uindex_t index) {
        if (index == n_halo_height_buffers) {
            return core_height;
        } else {
            return halo_height;
        }
    }

    /**
     * \brief Create a new IOKernel instance.
     *
     * The created instance is not invocable. You need to construct it inside a lambda function (or
     * equivalent) and then call either \ref IOKernel.read or \ref IOKernel.write.
     *
     * \param accessor The slice/column of buffer accessors to process.
     * \param n_columns The width of every buffer passed in `accessor`.
     */
    IOKernel(std::array<Accessor, n_buffers> accessor, uindex_t n_columns)
        : accessor(accessor), n_columns(n_columns) {
#ifndef __SYCL_DEVICE_ONLY__
        for (uindex_min_t i = 0; i < n_buffers; i++) {
            assert(get_buffer_height(i) * n_columns <=
                   accessor[i].get_range()[0] * burst_buffer_length);
        }
#endif
    }

    /**
     * \brief Read the cells from the buffers and write them to the pipe.
     */
    void read() {
        static_assert(access_mode == cl::sycl::access::mode::read ||
                      access_mode == cl::sycl::access::mode::read_write);

        run<>([](Accessor &accessor, uindex_2d_t burst_i, uindex_min_t cell_i) {
            pipe::write(accessor[burst_i][cell_i]);
        });
    }

    /**
     * \brief Read the cells from the pipe and write them to the buffers.
     */
    void write() {
        static_assert(access_mode == cl::sycl::access::mode::write ||
                      access_mode == cl::sycl::access::mode::discard_write ||
                      access_mode == cl::sycl::access::mode::read_write ||
                      access_mode == cl::sycl::access::mode::discard_read_write);
        run([](Accessor &accessor, uindex_2d_t i) {
            accessor[i] = pipe::read();
        });
    }

  private:
    template <typename Action> void run(Action action) {
        static_assert(std::is_invocable<Action, Accessor &, uindex_2d_t>::value);

        uindex_2d_t i[n_buffers] = {0};

        for (uindex_1d_t c = 0; c < n_columns; c++) {
            uindex_min_t buffer_i = 0;
            uindex_1d_t next_bound = get_buffer_height(0);
            for (uindex_1d_t r = 0; r < n_rows; r++) {
                if (r == next_bound) {
                    buffer_i++;
                    next_bound += get_buffer_height(buffer_i);
                }
            }
        }
    }

    [[intel::fpga_register]] std::array<Accessor, n_buffers> accessor;
    uindex_1d_t n_columns;
};

} // namespace tiling
} // namespace stencil