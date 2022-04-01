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
#include <CL/sycl.hpp>
#include <StencilStream/GenericID.hpp>
#include <StencilStream/tiling/Tile.hpp>
#include <ext/intel/fpga_extensions.hpp>
#include <res/catch.hpp>
#include <res/constants.hpp>

using namespace stencil;
using namespace cl::sycl;
using namespace stencil::tiling;
using namespace std;

using TileImpl = Tile<ID, tile_width, tile_height, halo_radius, 64>;
constexpr uindex_t burst_length = 8;
static_assert(burst_length == TileImpl::burst_buffer_length);

TEST_CASE("Tile::copy_from/Tile::copy_to", "[Tile]") {
    cl::sycl::buffer<ID, 2> in_buffer(cl::sycl::range<2>(grid_width, grid_height));
    cl::sycl::buffer<ID, 2> out_buffer(cl::sycl::range<2>(grid_width, grid_height));

    REQUIRE(grid_width > 2 * tile_width);
    REQUIRE(grid_height > 2 * tile_height);

    {
        auto in_ac = in_buffer.get_access<access::mode::discard_write>();
        auto out_ac = out_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                in_ac[c][r] = ID(c, r);
                out_ac[c][r] = ID(-1, -1);
            }
        }
    }

    TileImpl tile(tile_width / 2);
    tile.copy_from(in_buffer, cl::sycl::id<2>(tile_width, tile_height));
    tile.copy_to(out_buffer, cl::sycl::id<2>(tile_width, tile_height));

    {
        auto out_ac = out_buffer.get_access<access::mode::read>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                if (c >= tile_width && c < 1.5 * tile_width && r >= tile_height &&
                    r < 2 * tile_height) {
                    REQUIRE(out_ac[c][r].c == c);
                    REQUIRE(out_ac[c][r].r == r);
                } else {
                    REQUIRE(out_ac[c][r].c == -1);
                    REQUIRE(out_ac[c][r].r == -1);
                }
            }
        }
    }
}

void run_tile_write_test(uindex_t width) {
    using pipe = cl::sycl::pipe<class tile_write_test_pipe, ID>;
    cl::sycl::queue queue(ext::intel::fpga_emulator_selector().select_device());
    TileImpl tile(width);

    queue.submit([&](cl::sycl::handler &cgh) {
        cgh.single_task<class tile_write_test_input>([=]() {
            for (uindex_t c = 0; c < width; c++) {
                for (uindex_t r = 0; r < halo_radius; r++) {
                    pipe::write(ID(c, r));
                }
            }
            for (uindex_t c = 0; c < width; c++) {
                for (uindex_t r = halo_radius; r < tile_height - halo_radius; r++) {
                    pipe::write(ID(c, r));
                }
            }
            for (uindex_t c = 0; c < width; c++) {
                for (uindex_t r = tile_height - halo_radius; r < tile_height; r++) {
                    pipe::write(ID(c, r));
                }
            }
        });
    });
    tile.template submit_write<pipe>(queue, TileImpl::Stripe::NORTH);
    tile.template submit_write<pipe>(queue, TileImpl::Stripe::CORE);
    tile.template submit_write<pipe>(queue, TileImpl::Stripe::SOUTH);

    cl::sycl::buffer<ID, 2> out_buffer(cl::sycl::range<2>(width, tile_height));
    tile.copy_to(out_buffer, cl::sycl::id<2>(0, 0));

    auto ac = out_buffer.get_access<cl::sycl::access::mode::read>();
    for (uindex_t c = 0; c < width; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            REQUIRE(ac[c][r].c == c);
            REQUIRE(ac[c][r].r == r);
        }
    }
}

TEST_CASE("Tile::submit_write", "[Tile]") {
    run_tile_write_test(tile_width);
    run_tile_write_test(halo_radius);
}

void run_tile_read_test(uindex_t width) {
    using pipe = cl::sycl::pipe<class tile_read_test_pipe, ID>;
    cl::sycl::queue queue(ext::intel::fpga_emulator_selector().select_device());

    cl::sycl::buffer<ID, 2> in_buffer(cl::sycl::range<2>(width, tile_height));
    {
        auto ac = in_buffer.get_access<cl::sycl::access::mode::discard_write>();
        for (uindex_t c = 0; c < width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                ac[c][r] = ID(c, r);
            }
        }
    }

    TileImpl tile(width);
    tile.copy_from(in_buffer, cl::sycl::id<2>(0, 0));

    // ==============
    // Full read test
    // ==============

    tile.submit_read<pipe>(queue, TileImpl::Stripe::NORTH, TileImpl::StripePart::FULL);
    tile.submit_read<pipe>(queue, TileImpl::Stripe::CORE, TileImpl::StripePart::FULL);
    tile.submit_read<pipe>(queue, TileImpl::Stripe::SOUTH, TileImpl::StripePart::FULL);

    cl::sycl::buffer<ID, 2> out_buffer(cl::sycl::range<2>(width, tile_height));
    queue.submit([&](cl::sycl::handler &cgh) {
        auto ac = out_buffer.get_access<cl::sycl::access::mode::discard_write>(cgh);

        cgh.single_task<class tile_full_read_test>([=]() {
            for (uindex_t c = 0; c < width; c++) {
                for (uindex_t r = 0; r < halo_radius; r++) {
                    ac[c][r] = pipe::read();
                }
            }
            for (uindex_t c = 0; c < width; c++) {
                for (uindex_t r = halo_radius; r < core_height + halo_radius; r++) {
                    ac[c][r] = pipe::read();
                }
            }
            for (uindex_t c = 0; c < width; c++) {
                for (uindex_t r = core_height + halo_radius; r < tile_height; r++) {
                    ac[c][r] = pipe::read();
                }
            }
        });
    });

    {
        auto ac = out_buffer.get_access<cl::sycl::access::mode::read>();
        for (uindex_t c = 0; c < width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                REQUIRE(ac[c][r].c == c);
                REQUIRE(ac[c][r].r == r);
            }
        }
    }

    // =================
    // Header read test
    // =================

    tile.submit_read<pipe>(queue, TileImpl::Stripe::NORTH, TileImpl::StripePart::HEADER);
    tile.submit_read<pipe>(queue, TileImpl::Stripe::CORE, TileImpl::StripePart::HEADER);
    tile.submit_read<pipe>(queue, TileImpl::Stripe::SOUTH, TileImpl::StripePart::HEADER);

    out_buffer = cl::sycl::range<2>(halo_radius, tile_height);
    queue.submit([&](cl::sycl::handler &cgh) {
        auto ac = out_buffer.get_access<cl::sycl::access::mode::discard_write>(cgh);

        cgh.single_task<class tile_header_read_test>([=]() {
            for (uindex_t c = 0; c < halo_radius; c++) {
                for (uindex_t r = 0; r < halo_radius; r++) {
                    ac[c][r] = pipe::read();
                }
            }
            for (uindex_t c = 0; c < halo_radius; c++) {
                for (uindex_t r = halo_radius; r < core_height + halo_radius; r++) {
                    ac[c][r] = pipe::read();
                }
            }
            for (uindex_t c = 0; c < halo_radius; c++) {
                for (uindex_t r = core_height + halo_radius; r < tile_height; r++) {
                    ac[c][r] = pipe::read();
                }
            }
        });
    });

    {
        auto ac = out_buffer.get_access<cl::sycl::access::mode::read>();
        for (uindex_t c = 0; c < halo_radius; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                REQUIRE(ac[c][r].c == c);
                REQUIRE(ac[c][r].r == r);
            }
        }
    }

    // ================
    // Footer read test
    // ================

    tile.submit_read<pipe>(queue, TileImpl::Stripe::NORTH, TileImpl::StripePart::FOOTER);
    tile.submit_read<pipe>(queue, TileImpl::Stripe::CORE, TileImpl::StripePart::FOOTER);
    tile.submit_read<pipe>(queue, TileImpl::Stripe::SOUTH, TileImpl::StripePart::FOOTER);

    queue.submit([&](cl::sycl::handler &cgh) {
        auto ac = out_buffer.get_access<cl::sycl::access::mode::discard_write>(cgh);

        cgh.single_task<class tile_footer_read_test>([=]() {
            for (uindex_t c = 0; c < halo_radius; c++) {
                for (uindex_t r = 0; r < halo_radius; r++) {
                    ac[c][r] = pipe::read();
                }
            }
            for (uindex_t c = 0; c < halo_radius; c++) {
                for (uindex_t r = halo_radius; r < core_height + halo_radius; r++) {
                    ac[c][r] = pipe::read();
                }
            }
            for (uindex_t c = 0; c < halo_radius; c++) {
                for (uindex_t r = core_height + halo_radius; r < tile_height; r++) {
                    ac[c][r] = pipe::read();
                }
            }
        });
    });

    {
        auto ac = out_buffer.get_access<cl::sycl::access::mode::read>();
        for (uindex_t c = 0; c < halo_radius; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                REQUIRE(ac[c][r].c == width - halo_radius + c);
                REQUIRE(ac[c][r].r == r);
            }
        }
    }
}

TEST_CASE("Tile::submit_read", "[Tile]") {
    run_tile_read_test(tile_width);
    run_tile_read_test(halo_radius);
}