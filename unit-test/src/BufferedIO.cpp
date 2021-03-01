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
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/range.hpp>
#include <StencilStream/BufferedIO.hpp>
#include <StencilStream/GenericID.hpp>

using namespace cl::sycl;
using namespace stencil_stream;

TEST_CASE("Buffered read-out is correct", "[BufferedIO]")
{
    constexpr UIndex flush_length = 256;
    constexpr UIndex n_blocks = 10;

    buffer<UID, 2> in_buffer(range<2>(n_blocks, flush_length));

    {
        auto ac = in_buffer.get_access<access::mode::discard_write>();
        for (UIndex block_i = 0; block_i < n_blocks; block_i++)
        {
            for (UIndex cell_i = 0; cell_i < flush_length; cell_i++)
            {
                ac[block_i][cell_i] = UID(block_i, cell_i);
            }
        }
    }

    BufferedIO<UID, flush_length, access::mode::read, access::target::host_buffer> reader(true, in_buffer.get_access<access::mode::read>());

    for (UIndex block_i = 0; block_i < n_blocks; block_i++)
    {
        for (UIndex cell_i = 0; cell_i < flush_length; cell_i++)
        {
            UID next_value = reader.read();
            REQUIRE(next_value.c == block_i);
            REQUIRE(next_value.r == cell_i);
        }
    }

    // Check that the overread behavior is good (the last two blocks are repeated).
    for (UIndex cell_i = 0; cell_i < flush_length; cell_i++)
    {
        UID next_value = reader.read();
        REQUIRE(next_value.c == n_blocks - 2);
        REQUIRE(next_value.r == cell_i);
    }

    for (UIndex cell_i = 0; cell_i < flush_length; cell_i++)
    {
        UID next_value = reader.read();
        REQUIRE(next_value.c == n_blocks - 1);
        REQUIRE(next_value.r == cell_i);
    }
}

TEST_CASE("Buffered write is correct", "[BufferedIO]")
{
    constexpr UIndex flush_length = 256;
    constexpr UIndex n_blocks = 10;

    buffer<UID, 2> out_buffer(range<2>(n_blocks, flush_length));

    {
        BufferedIO<UID, flush_length, access::mode::discard_write, access::target::host_buffer> writer(false, out_buffer.get_access<access::mode::discard_write>());

        // Also writing a little bit to much to test the behavior with superflous writes.
        for (UIndex block_i = 0; block_i < n_blocks+2; block_i++)
        {
            for (UIndex cell_i = 0; cell_i < flush_length; cell_i++)
            {
                writer.write(UID(block_i, cell_i));
            }
        }

        // We've written full blocks, no flushing necessary.
    }

    auto ac = out_buffer.get_access<access::mode::read>();
    for (UIndex block_i = 0; block_i < n_blocks; block_i++)
    {
        for (UIndex cell_i = 0; cell_i < flush_length; cell_i++)
        {
            UID next_value = ac[block_i][cell_i];
            REQUIRE(next_value.c == block_i);
            REQUIRE(next_value.r == cell_i);
        }
    }
}

TEST_CASE("Flushing is correct", "[BufferedIO]")
{
    constexpr UIndex flush_length = 256;
    constexpr UIndex n_modulo_cells = 100;
    constexpr UIndex n_blocks = 10;

    buffer<UID, 2> out_buffer(range<2>(n_blocks, flush_length));

    {
        BufferedIO<UID, flush_length, access::mode::discard_write, access::target::host_buffer> writer(false, out_buffer.get_access<access::mode::discard_write>());

        for (UIndex block_i = 0; block_i < n_blocks - 1; block_i++)
        {
            for (UIndex cell_i = 0; cell_i < flush_length; cell_i++)
            {
                writer.write(UID(block_i, cell_i));
            }
        }

        for (UIndex cell_i = 0; cell_i < n_modulo_cells; cell_i++)
        {
            writer.write(UID(n_blocks-1, cell_i));
        }

        writer.flush();
    }

    auto ac = out_buffer.get_access<access::mode::read>();
    for (UIndex block_i = 0; block_i < n_blocks - 1; block_i++)
    {
        for (UIndex cell_i = 0; cell_i < flush_length; cell_i++)
        {
            UID next_value = ac[block_i][cell_i];
            REQUIRE(next_value.c == block_i);
            REQUIRE(next_value.r == cell_i);
        }
    }
    for (UIndex cell_i = 0; cell_i < n_modulo_cells; cell_i++)
    {
        UID next_value = ac[n_blocks-1][cell_i];
        REQUIRE(next_value.c == n_blocks-1);
        REQUIRE(next_value.r == cell_i);
    }
}