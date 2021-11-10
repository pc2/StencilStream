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
#include <res/HostPipe.hpp>
#include <res/catch.hpp>
#include <res/constants.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/range.hpp>
#include <StencilStream/GenericID.hpp>
#include <StencilStream/tiling/IOKernel.hpp>

using namespace cl::sycl;
using namespace stencil::tiling;
using namespace stencil;
using namespace std;

TEST_CASE("IOKernel::read()", "[IOKernel]") {
    buffer<UID, 1> in_buffer[5] = {buffer<UID, 1>(range<1>(halo_radius * halo_radius)),
                                   buffer<UID, 1>(range<1>(halo_radius * halo_radius)),
                                   buffer<UID, 1>(range<1>(halo_radius * core_height)),
                                   buffer<UID, 1>(range<1>(halo_radius * halo_radius)),
                                   buffer<UID, 1>(range<1>(halo_radius * halo_radius))};

    {
        accessor<UID, 1, access::mode::discard_write, access::target::host_buffer> in_buffer_ac[5] =
            {in_buffer[0].get_access<access::mode::discard_write>(),
             in_buffer[1].get_access<access::mode::discard_write>(),
             in_buffer[2].get_access<access::mode::discard_write>(),
             in_buffer[3].get_access<access::mode::discard_write>(),
             in_buffer[4].get_access<access::mode::discard_write>()};

        for (uindex_t c = 0; c < halo_radius; c++) {
            for (uindex_t r = 0; r < halo_radius; r++) {
                in_buffer_ac[0][c*halo_radius + r] = UID(c, r);
                in_buffer_ac[1][c*halo_radius + r] = UID(c, r);
                in_buffer_ac[3][c*halo_radius + r] = UID(c, r);
                in_buffer_ac[4][c*halo_radius + r] = UID(c, r);
            }
        }

        for (uindex_t c = 0; c < halo_radius; c++) {
            for (uindex_t r = 0; r < core_height; r++) {
                in_buffer_ac[2][c*core_height + r] = UID(c, r);
            }
        }
    }

    using in_pipe = HostPipe<class BufferedInputKernelPipeID, UID>;
    using InputKernel = IOKernel<UID, halo_radius, core_height, in_pipe, 2, access::mode::read, access::target::host_buffer>;

    {
        array<InputKernel::Accessor, 5> accessors = {in_buffer[0].get_access<access::mode::read>(),
                                                     in_buffer[1].get_access<access::mode::read>(),
                                                     in_buffer[2].get_access<access::mode::read>(),
                                                     in_buffer[3].get_access<access::mode::read>(),
                                                     in_buffer[4].get_access<access::mode::read>()};

        InputKernel kernel(accessors, halo_radius);
        kernel.read();
    };

    buffer<UID, 2> out_buffer(range<2>(halo_radius, 2 * halo_radius + tile_height));
    auto out_buffer_ac = out_buffer.get_access<access::mode::read_write>();
    for (uindex_t c = 0; c < halo_radius; c++) {
        for (uindex_t r = 0; r < 2 * halo_radius + tile_height; r++) {
            out_buffer_ac[c][r] = in_pipe::read();
        }
    }
    REQUIRE(in_pipe::empty());

    for (uindex_t c = 0; c < halo_radius; c++) {
        for (uindex_t r = 0; r < halo_radius; r++) {
            REQUIRE(out_buffer_ac[c][r].c == c);
            REQUIRE(out_buffer_ac[c][r].r == r);
            REQUIRE(out_buffer_ac[c][r + halo_radius].c == c);
            REQUIRE(out_buffer_ac[c][r + halo_radius].r == r);
            REQUIRE(out_buffer_ac[c][r + 2 * halo_radius + core_height].c == c);
            REQUIRE(out_buffer_ac[c][r + 2 * halo_radius + core_height].r == r);
            REQUIRE(out_buffer_ac[c][r + 3 * halo_radius + core_height].c == c);
            REQUIRE(out_buffer_ac[c][r + 3 * halo_radius + core_height].r == r);
        }
    }

    for (uindex_t c = 0; c < halo_radius; c++) {
        for (uindex_t r = 0; r < core_height; r++) {
            REQUIRE(out_buffer_ac[c][r + 2 * halo_radius].c == c);
            REQUIRE(out_buffer_ac[c][r + 2 * halo_radius].r == r);
        }
    }
}

TEST_CASE("IOKernel::write()", "[IOKernel]") {
    buffer<UID, 2> in_buffer(range<2>(halo_radius, tile_height));
    auto in_buffer_ac = in_buffer.get_access<access::mode::read_write>();

    using out_pipe = HostPipe<class BufferedOutputKernelPipeID, UID>;

    for (uindex_t c = 0; c < halo_radius; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            if (r < halo_radius) {
                out_pipe::write(UID(c, r));
            } else if (r < halo_radius + core_height) {
                out_pipe::write(UID(c, r - halo_radius));
            } else {
                out_pipe::write(UID(c, r - halo_radius - core_height));
            }
        }
    }

    buffer<UID, 1> out_buffer[3] = {buffer<UID, 1>(range<1>(halo_radius * halo_radius)),
                                    buffer<UID, 1>(range<1>(halo_radius * core_height)),
                                    buffer<UID, 1>(range<1>(halo_radius * halo_radius))};

    using OutputKernel = IOKernel<UID, halo_radius, core_height, out_pipe, 1, access::mode::read_write, access::target::host_buffer>;

    array<OutputKernel::Accessor, 3> out_buffer_ac = {
        out_buffer[0].get_access<access::mode::read_write>(),
        out_buffer[1].get_access<access::mode::read_write>(),
        out_buffer[2].get_access<access::mode::read_write>()};

    OutputKernel kernel(out_buffer_ac, halo_radius);
    kernel.write();

    REQUIRE(out_pipe::empty());

    for (uindex_t c = 0; c < halo_radius; c++) {
        for (uindex_t r = 0; r < halo_radius; r++) {
            REQUIRE(out_buffer_ac[0][c * halo_radius + r].c == c);
            REQUIRE(out_buffer_ac[0][c * halo_radius + r].r == r);
            REQUIRE(out_buffer_ac[2][c * halo_radius + r].c == c);
            REQUIRE(out_buffer_ac[2][c * halo_radius + r].r == r);
        }
    }

    for (uindex_t c = 0; c < halo_radius; c++) {
        for (uindex_t r = 0; r < core_height; r++) {
            REQUIRE(out_buffer_ac[1][c * core_height + r].c == c);
            REQUIRE(out_buffer_ac[1][c * core_height + r].r == r);
        }
    }
}