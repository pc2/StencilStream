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
#include "Index.hpp"
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/id.hpp>

namespace stencil
{
inline cl::sycl::range<2> burst_partitioned_range(uindex_t width, uindex_t height, uindex_t burst_length)
{
    uindex_t nCells = width * height;
    cl::sycl::range<2> range(nCells / burst_length, burst_length);
    if (nCells % burst_length != 0)
    {
        range[0] += 1;
    }
    return range;
}

inline constexpr bool is_mode_readable(cl::sycl::access::mode access_mode)
{
    return access_mode == cl::sycl::access::mode::read ||
           access_mode == cl::sycl::access::mode::read_write;
}

inline constexpr bool is_mode_writable(cl::sycl::access::mode access_mode)
{
    return access_mode == cl::sycl::access::mode::write ||
           access_mode == cl::sycl::access::mode::read_write ||
           access_mode == cl::sycl::access::mode::discard_write ||
           access_mode == cl::sycl::access::mode::discard_read_write;
}
} // namespace stencil