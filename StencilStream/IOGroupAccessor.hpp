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
#include "BufferedIO.hpp"
#include "Index.hpp"
#include <optional>

namespace stencil_stream
{
template <typename T, UIndex halo_height, UIndex core_height, UIndex flush_length, cl::sycl::access::mode access_mode, cl::sycl::access::target access_target>
class IOGroupAccessor
{
    using Buffer = BufferedIO<T, flush_length, access_mode, access_target>;
    using Accessor = typename Buffer::Accessor;

public:
    IOGroupAccessor(
        Accessor upper_northern_halo,
        Accessor lower_northern_halo,
        Accessor core,
        Accessor upper_southern_halo,
        Accessor lower_southern_halo) : upper_northern_halo(Buffer(true, upper_northern_halo)),
                                        lower_northern_halo(true, lower_northern_halo),
                                        core(true, core),
                                        upper_southern_halo(true, upper_southern_halo),
                                        lower_southern_halo(Buffer(true, lower_southern_halo)),
                                        i_row(0)
    {
        static_assert(access_mode == cl::sycl::access::mode::read || access_mode == cl::sycl::access::mode::read_write);
    }

    IOGroupAccessor(
        Accessor northern_halo,
        Accessor core,
        Accessor southern_halo) : upper_northern_halo(std::nullopt),
                                  lower_northern_halo(false, northern_halo),
                                  core(false, core),
                                  upper_southern_halo(false, southern_halo),
                                  lower_southern_halo(std::nullopt),
                                  i_row(0)
    {
        static_assert(access_mode == cl::sycl::access::mode::write || access_mode == cl::sycl::access::mode::read_write || access_mode == cl::sycl::access::mode::discard_write || access_mode == cl::sycl::access::mode::discard_read_write);
    }

    T read()
    {
        T new_value;

        if (i_row < halo_height)
        {
            new_value = upper_northern_halo->read();
        }
        else if (i_row < 2 * halo_height)
        {
            new_value = lower_northern_halo.read();
        }
        else if (i_row < 2 * halo_height + core_height)
        {
            new_value = core.read();
        }
        else if (i_row < 3 * halo_height + core_height)
        {
            new_value = upper_southern_halo.read();
        }
        else
        {
            new_value = lower_southern_halo->read();
        }

        step_index();

        return new_value;
    }

    void write(T new_value)
    {
        if (i_row < halo_height)
        {
            lower_northern_halo.write(new_value);
        }
        else if (i_row < 2 * halo_height)
        {
            core.write(new_value);
        }
        else
        {
            upper_southern_halo.write(new_value);
        }

        step_index();
    }

    void flush()
    {
        lower_northern_halo.flush();
        core.flush();
        upper_southern_halo.flush();
    }

private:
    void step_index()
    {
        if (i_row == 4 * halo_height + core_height - 1)
        {
            i_row = 0;
        }
        else
        {
            i_row++;
        }
    }

    std::optional<Buffer> upper_northern_halo;
    Buffer lower_northern_halo, core, upper_southern_halo;
    std::optional<Buffer> lower_southern_halo;

    UIndex i_row;
};
} // namespace stencil_stream