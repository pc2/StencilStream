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

constexpr uindex_t burst_buffer_length = 2;
constexpr uindex_t corner_bursts = halo_radius * halo_radius / burst_buffer_length;
constexpr uindex_t edge_bursts = halo_radius * core_height / burst_buffer_length;

TEST_CASE("IOKernel::read()", "[IOKernel]") {
    buffer<UID[burst_buffer_length], 1> in_buffer[5] = {
        buffer<UID[burst_buffer_length], 1>(range<1>(corner_bursts)),
        buffer<UID[burst_buffer_length], 1>(range<1>(corner_bursts)),
        buffer<UID[burst_buffer_length], 1>(range<1>(edge_bursts)),
        buffer<UID[burst_buffer_length], 1>(range<1>(corner_bursts)),
        buffer<UID[burst_buffer_length], 1>(range<1>(corner_bursts))};

    {
        accessor<UID[burst_buffer_length], 1, access::mode::discard_write, access::target::host_buffer> in_buffer_ac[5] =
            {in_buffer[0].get_access<access::mode::discard_write>(),
             in_buffer[1].get_access<access::mode::discard_write>(),
             in_buffer[2].get_access<access::mode::discard_write>(),
             in_buffer[3].get_access<access::mode::discard_write>(),
             in_buffer[4].get_access<access::mode::discard_write>()};

        for (uindex_t buffer_i = 0; buffer_i < 5; buffer_i++) {
            uindex_t n_bursts;
            if (buffer_i == 2) {
                n_bursts = edge_bursts;
            } else {
                n_bursts = corner_bursts;
            }

            for (uindex_t burst_i = 0; burst_i < n_bursts; burst_i++) {
                for (uindex_t cell_i = 0; cell_i < burst_buffer_length; cell_i++) {
                    in_buffer_ac[buffer_i][burst_i][cell_i]
                        = UID(burst_i, cell_i);
                }
            }
        }
    }

    using in_pipe = HostPipe<class BufferedInputKernelPipeID, UID>;
    using InputKernel = IOKernel<UID, halo_radius, core_height, in_pipe, 2, access::mode::read, access::target::host_buffer, burst_buffer_length>;

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
            uindex_t burst_i = (c * halo_radius + r) / burst_buffer_length;
            uindex_t cell_i = (c * halo_radius + r) % burst_buffer_length;

            REQUIRE(out_buffer_ac[c][r].c == burst_i);
            REQUIRE(out_buffer_ac[c][r].r == cell_i);
            REQUIRE(out_buffer_ac[c][r + halo_radius].c == burst_i);
            REQUIRE(out_buffer_ac[c][r + halo_radius].r == cell_i);
            REQUIRE(out_buffer_ac[c][r + 2 * halo_radius + core_height].c == burst_i);
            REQUIRE(out_buffer_ac[c][r + 2 * halo_radius + core_height].r == cell_i);
            REQUIRE(out_buffer_ac[c][r + 3 * halo_radius + core_height].c == burst_i);
            REQUIRE(out_buffer_ac[c][r + 3 * halo_radius + core_height].r == cell_i);
        }
    }

    for (uindex_t c = 0; c < halo_radius; c++) {
        for (uindex_t r = 0; r < core_height; r++) {
            uindex_t burst_i = (c * core_height + r) / burst_buffer_length;
            uindex_t cell_i = (c * core_height + r) % burst_buffer_length;

            REQUIRE(out_buffer_ac[c][r + 2 * halo_radius].c == burst_i);
            REQUIRE(out_buffer_ac[c][r + 2 * halo_radius].r == cell_i);
        }
    }
}

TEST_CASE("IOKernel::write()", "[IOKernel]") {
    buffer<UID, 2> in_buffer(range<2>(halo_radius, tile_height));
    auto in_buffer_ac = in_buffer.get_access<access::mode::read_write>();

    using out_pipe = HostPipe<class BufferedOutputKernelPipeID, UID>;

    for (uindex_t c = 0; c < halo_radius; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            uindex_t i;
            if (r < halo_radius) {
                i = c * halo_radius + r;
            } else if (r < halo_radius + core_height) {
                i = c * core_height + r - halo_radius;
            } else {
                i = c * halo_radius + r - halo_radius - core_height;
            }
            uindex_t burst_i = i / burst_buffer_length;
            uindex_t cell_i = i % burst_buffer_length;
            out_pipe::write(UID(burst_i, cell_i));
        }
    }

    buffer<UID[burst_buffer_length], 1> out_buffer[3] = {buffer<UID[burst_buffer_length], 1>(range<1>(corner_bursts)),
                                    buffer<UID[burst_buffer_length], 1>(range<1>(edge_bursts)),
                                    buffer<UID[burst_buffer_length], 1>(range<1>(corner_bursts))};

    using OutputKernel = IOKernel<UID, halo_radius, core_height, out_pipe, 1, access::mode::read_write, access::target::host_buffer, burst_buffer_length>;

    array<OutputKernel::Accessor, 3> out_buffer_ac = {
        out_buffer[0].get_access<access::mode::read_write>(),
        out_buffer[1].get_access<access::mode::read_write>(),
        out_buffer[2].get_access<access::mode::read_write>()};

    OutputKernel kernel(out_buffer_ac, halo_radius);
    kernel.write();

    REQUIRE(out_pipe::empty());

    for (uindex_t buffer_i = 0; buffer_i < 3; buffer_i++) {
        uindex_t n_bursts;
        uindex_t height;
        if (buffer_i == 1) {
            n_bursts = edge_bursts;
            height = core_height;
        } else {
            n_bursts = corner_bursts;
            height = halo_radius;
        }

        for (uindex_t burst_i = 0; burst_i < n_bursts; burst_i++) {
            for (uindex_t cell_i = 0; cell_i < burst_buffer_length; cell_i++) {
                REQUIRE(out_buffer_ac[buffer_i][burst_i][cell_i].c == burst_i);
                REQUIRE(out_buffer_ac[buffer_i][burst_i][cell_i].r == cell_i);
            }
        }
    }
}
