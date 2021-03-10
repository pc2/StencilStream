/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "catch.hpp"
#include <StencilStream/Helpers.hpp>
#include <StencilStream/IOGroupAccessor.hpp>

using namespace cl::sycl;
using namespace std;
using namespace stencil_stream;

constexpr UIndex halo_height = 64;
constexpr UIndex core_height = 512;
constexpr UIndex width = halo_height;
constexpr UIndex flush_length = 256;

range<2> halo_range = flushblocked_buffer_range(width, halo_height, flush_length);
range<2> core_range = flushblocked_buffer_range(width, core_height, flush_length);

TEST_CASE("Reading works", "[IOGroupAccessor]")
{
    buffer<UID, 2> upper_northern_halo(halo_range);
    buffer<UID, 2> lower_northern_halo(halo_range);
    buffer<UID, 2> core(core_range);
    buffer<UID, 2> upper_southern_halo(halo_range);
    buffer<UID, 2> lower_southern_halo(halo_range);

    // TODO: Write valid data into the buffers.

    {
        auto unh_ac = upper_northern_halo.get_access<access::mode::read>();
        auto lnh_ac = lower_northern_halo.get_access<access::mode::read>();
        auto core_ac = core.get_access<access::mode::read>();
        auto ush_ac = upper_southern_halo.get_access<access::mode::read>();
        auto lsh_ac = lower_southern_halo.get_access<access::mode::read>();

        IOGroupAccessor<UID, halo_height, core_height, flush_length, access::mode::read, access::target::host_buffer> group_ac(unh_ac, lnh_ac, core_ac, ush_ac, lsh_ac);

        for (UIndex c = 0; c < width; c++)
        {
            for (UIndex r = 0; r < 4 * halo_height + core_height; r++)
            {
                UID read_value = group_ac.read();
                REQUIRE(read_value == UID(c, r));
            }
        }
    }
}

TEST_CASE("Writing works", "[IOGroupAccessor]")
{
    buffer<UID, 2> northern_halo(halo_range);
    buffer<UID, 2> core(core_range);
    buffer<UID, 2> southern_halo(halo_range);

    {
        auto nh_ac = northern_halo.get_access<access::mode::discard_write>();
        auto core_ac = core.get_access<access::mode::discard_write>();
        auto sh_ac = southern_halo.get_access<access::mode::discard_write>();

        IOGroupAccessor<UID, halo_height, core_height, flush_length, access::mode::discard_write, access::target::host_buffer> group_ac(nh_ac, core_ac, sh_ac);

        for (UIndex c = 0; c < width; c++)
        {
            for (UIndex r = 0; r < 2 * halo_height + core_height; r++)
            {
                group_ac.write(UID(c, r));
            }
        }

        group_ac.flush();
    }

    // TODO: Validate written data.
}