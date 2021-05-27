/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include "CounterID.hpp"
#include "Index.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/accessor.hpp>
#include <array>

namespace stencil
{

template <typename T, uindex_t halo_radius, uindex_t core_height, uindex_t burst_length, typename pipe, uindex_t n_halo_buffers, cl::sycl::access::mode access_mode, cl::sycl::access::target access_target = cl::sycl::access::target::global_buffer>
class IOKernel
{
public:
    using Accessor = cl::sycl::accessor<T, 2, access_mode, access_target>;

    static constexpr uindex_t n_buffers = 2 * n_halo_buffers + 1;
    static constexpr uindex_t n_rows = 2 * n_halo_buffers * halo_radius + core_height;

    static constexpr uindex_t get_buffer_height(uindex_t index)
    {
        if (index == n_halo_buffers)
        {
            return core_height;
        }
        else
        {
            return halo_radius;
        }
    }

    IOKernel(std::array<Accessor, n_buffers> accessor, uindex_t n_columns) : accessor(accessor), n_columns(n_columns)
    {
#ifndef __SYCL_DEVICE_ONLY__
        for (uindex_t i = 0; i < n_buffers; i++)
        {
            assert(accessor[i].get_range()[1] == burst_length);
            assert(get_buffer_height(i) * n_columns <= accessor[i].get_range()[0] * accessor[i].get_range()[1]);
        }
#endif
    }

    void read()
    {
        static_assert(access_mode == cl::sycl::access::mode::read || access_mode == cl::sycl::access::mode::read_write);
        run([](Accessor &accessor, uindex_t burst_i, uindex_t cell_i) {
            pipe::write(accessor[burst_i][cell_i]);
        });
    }

    void write()
    {
        static_assert(access_mode == cl::sycl::access::mode::write || access_mode == cl::sycl::access::mode::discard_write || access_mode == cl::sycl::access::mode::read_write || access_mode == cl::sycl::access::mode::discard_read_write);
        run([](Accessor &accessor, uindex_t burst_i, uindex_t cell_i) {
            accessor[burst_i][cell_i] = pipe::read();
        });
    }

private:
    template <typename Action>
    void run(Action action)
    {
        static_assert(std::is_invocable<Action, Accessor &, uindex_t, uindex_t>::value);

        uindex_t burst_i[n_buffers] = {0};
        uindex_t cell_i[n_buffers] = {0};

        for (uindex_t c = 0; c < n_columns; c++)
        {
            uindex_t buffer_i = 0;
            uindex_t next_bound = get_buffer_height(0);
            for (uindex_t r = 0; r < n_rows; r++)
            {
                if (r == next_bound)
                {
                    buffer_i++;
                    next_bound += get_buffer_height(buffer_i);
                }

                action(accessor[buffer_i], burst_i[buffer_i], cell_i[buffer_i]);
                if (cell_i[buffer_i] == burst_length - 1)
                {
                    cell_i[buffer_i] = 0;
                    burst_i[buffer_i]++;
                }
                else
                {
                    cell_i[buffer_i]++;
                }
            }
        }
    }

    std::array<Accessor, n_buffers> accessor;
    uindex_t n_columns;
};

} // namespace stencil