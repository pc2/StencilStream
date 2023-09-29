/*
 * Copyright © 2020-2023 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "../constants.hpp"
#include <CL/sycl.hpp>
#include <StencilStream/GenericID.hpp>
#include <StencilStream/tiling/Tile.hpp>
#include <catch2/catch_all.hpp>

using namespace stencil;
using namespace sycl;
using namespace stencil::tiling;
using namespace std;

using TileImpl = Tile<ID, tile_width, tile_height, halo_radius, 64>;
constexpr uindex_t word_length = 8;
static_assert(word_length == TileImpl::word_length);

TEST_CASE("Tile::get_part_buffer", "[Tile]") {
    TileImpl tile;

    for (TileImpl::Part part_type : TileImpl::all_parts) {
        auto part = tile.get_part_buffer(part_type);

        uindex_t required_words = TileImpl::get_part_words(part_type);
        REQUIRE(required_words == part.get_range()[0]);
    }
}

TEST_CASE("Tile::TileAccessor", "[Tile]") {
    TileImpl tile;

    {
        TileImpl::TileAccessor<access::mode::read_write> tile_ac(tile);
        for (uindex_t c = 0; c < tile_width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                tile_ac[c][r] = ID(c, r);
            }
        }
    }

    for (TileImpl::Part part : TileImpl::all_parts) {
        auto true_range = TileImpl::get_part_range(part);
        auto content_offset = TileImpl::get_part_offset(part);
        auto part_buffer = tile.get_part_buffer(part);
        host_accessor part_ac(part_buffer, read_only);

        for (uindex_t c = 0; c < true_range.c; c++) {
            for (uindex_t r = 0; r < true_range.r; r++) {
                if (c + content_offset[0] >= tile_width) {
                    continue;
                }
                if (r + content_offset[1] >= tile_height) {
                    continue;
                }
                uindex_t word_i = (c * true_range.r + r) / word_length;
                uindex_t cell_i = (c * true_range.r + r) % word_length;
                REQUIRE(part_ac[word_i][cell_i].value.c == c + content_offset[0]);
                REQUIRE(part_ac[word_i][cell_i].value.r == r + content_offset[1]);
            }
        }
    }

    {
        TileImpl::TileAccessor<access::mode::read> tile_ac(tile);
        for (uindex_t c = 0; c < tile_width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                REQUIRE(tile_ac[c][r] == ID(c, r));
            }
        }
    }
}

void copy_from_test_impl(uindex_t tile_width, uindex_t tile_height) {
    buffer<ID, 2> in_buffer(range<2>(tile_width, tile_height));

    {
        host_accessor in_buffer_ac(in_buffer, write_only);
        for (uindex_t c = 0; c < tile_width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                in_buffer_ac[c][r] = ID(c, r);
            }
        }
    }

    TileImpl tile;
    tile.copy_from(in_buffer, id<2>(0, 0));

    {
        TileImpl::TileAccessor<access::mode::read> tile_ac(tile);
        for (uindex_t c = 0; c < tile_width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                REQUIRE(tile_ac[c][r] == ID(c, r));
            }
        }
    }
}

TEST_CASE("Tile::copy_from", "[Tile]") {
    // Test for the full tile.
    copy_from_test_impl(tile_width, tile_height);
    // Test for a partial buffer.
    copy_from_test_impl(tile_width - 2 * halo_radius, tile_height - 2 * halo_radius);
}

void copy_to_test_impl(uindex_t tile_width, uindex_t tile_height) {
    TileImpl tile;

    {
        TileImpl::TileAccessor<access::mode::read_write> tile_ac(tile);
        for (uindex_t c = 0; c < tile_width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                tile_ac[c][r] = ID(c, r);
            }
        }
    }

    buffer<ID, 2> out_buffer(range<2>(tile_width, tile_height));
    tile.copy_to(out_buffer, id<2>(0, 0));

    host_accessor out_buffer_ac(out_buffer, read_only);
    for (uindex_t c = 0; c < tile_width; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            REQUIRE(out_buffer_ac[c][r].c == c);
            REQUIRE(out_buffer_ac[c][r].r == r);
        }
    }
}

TEST_CASE("Tile::copy_to", "[Tile]") {
    // Test for the full tile.
    copy_to_test_impl(tile_width, tile_height);
    // Test with a partial buffer.
    copy_to_test_impl(tile_width - 2 * halo_radius, tile_height - 2 * halo_radius);
}

void submit_read_part_test_impl(TileImpl::Part part, uindex_t n_columns) {
    uindex_t n_rows = TileImpl::get_part_range(part).r;

    TileImpl tile;
    {
        TileImpl::TileAccessor<access::mode::read_write> tile_ac(tile);
        for (uindex_t c = 0; c < tile_width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                tile_ac[c][r] = ID(c, r);
            }
        }
    }

    using in_pipe = sycl::pipe<class submit_read_part_test_pipe_id, ID>;
    queue queue;
    tile.submit_read_part<in_pipe>(queue, part, n_columns);

    buffer<ID, 2> out_buffer = range<2>(n_columns, n_rows);
    queue.submit([&](handler &cgh) {
        accessor ac(out_buffer, cgh, write_only);

        cgh.single_task([=]() {
            for (uindex_t c = 0; c < n_columns; c++) {
                for (uindex_t r = 0; r < n_rows; r++) {
                    ac[c][r] = in_pipe::read();
                }
            }
        });
    });

    host_accessor out_buffer_ac(out_buffer, read_only);
    id<2> part_offset = TileImpl::get_part_offset(part);
    for (uindex_t c = 0; c < n_columns; c++) {
        for (uindex_t r = 0; r < n_rows; r++) {
            REQUIRE(out_buffer_ac[c][r].c == part_offset[0] + c);
            REQUIRE(out_buffer_ac[c][r].r == part_offset[1] + r);
        }
    }
}

TEST_CASE("Tile::submit_read_part", "[Tile]") {
    for (auto part : TileImpl::all_parts) {
        UID range = TileImpl::get_part_range(part);
        submit_read_part_test_impl(part, range.c);
        submit_read_part_test_impl(part, range.c - 1);
    }
}

void submit_write_part_test_impl(TileImpl::Part part, uindex_t n_columns) {
    using out_pipe = sycl::pipe<class submit_read_part_test_pipe_id, ID>;
    queue queue;
    uindex_t n_rows = TileImpl::get_part_range(part).r;
    id<2> part_offset = TileImpl::get_part_offset(part);

    queue.submit([&](handler &cgh) {
        cgh.single_task([=]() {
            for (uindex_t c = 0; c < n_columns; c++) {
                for (uindex_t r = 0; r < n_rows; r++) {
                    out_pipe::write(ID(part_offset[0] + c, part_offset[1] + r));
                }
            }
        });
    });

    TileImpl tile;
    tile.submit_write_part<out_pipe>(queue, part, n_columns);

    TileImpl::TileAccessor<access::mode::read> tile_ac(tile);
    for (uindex_t c = 0; c < n_columns; c++) {
        for (uindex_t r = 0; r < n_rows; r++) {
            uindex_t global_c = part_offset[0] + c;
            uindex_t global_r = part_offset[1] + r;
            ID cell = tile_ac[global_c][global_r];
            REQUIRE(cell.c == global_c);
            REQUIRE(cell.r == global_r);
        }
    }
}

TEST_CASE("Tile::submit_write_part", "[Tile]") {
    for (auto part : TileImpl::all_parts) {
        UID range = TileImpl::get_part_range(part);
        submit_write_part_test_impl(part, range.c);
        submit_write_part_test_impl(part, range.c - 1);
    }
}
