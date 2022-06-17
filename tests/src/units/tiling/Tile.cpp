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
#include <res/catch.hpp>
#include <res/constants.hpp>

using namespace stencil;
using namespace cl::sycl;
using namespace stencil::tiling;
using namespace std;

using TileImpl = Tile<ID, tile_width, tile_height, halo_radius, 64>;
constexpr uindex_t word_length = 8;
static_assert(word_length == TileImpl::word_length);

TEST_CASE("Tile::operator[]", "[Tile]") {
    TileImpl tile;

    for (TileImpl::Part part_type : TileImpl::all_parts) {
        auto part = tile[part_type];

        uindex_t required_words = TileImpl::get_part_words(part_type);
        REQUIRE(required_words == part.get_range()[0]);
    }
}

void copy_from_test_impl(uindex_t tile_width, uindex_t tile_height) {
    buffer<ID, 2> in_buffer(range<2>(tile_width, tile_height));

    {
        auto in_buffer_ac = in_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < tile_width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                in_buffer_ac[c][r] = ID(c, r);
            }
        }
    }

    TileImpl tile;
    tile.copy_from(in_buffer, id<2>(0, 0));

    for (TileImpl::Part part : TileImpl::all_parts) {
        auto true_range = TileImpl::get_part_range(part);
        auto content_offset = TileImpl::get_part_offset(part);
        auto part_ac = tile[part].get_access<access::mode::read>();

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
}

TEST_CASE("Tile::copy_from", "[Tile]") {
    // Test for the full tile.
    copy_from_test_impl(tile_width, tile_height);
    // Test for a partial buffer.
    copy_from_test_impl(tile_width - 2 * halo_radius, tile_height - 2 * halo_radius);
}

void copy_to_test_impl(uindex_t tile_width, uindex_t tile_height) {
    TileImpl tile;

    for (TileImpl::Part part : TileImpl::all_parts) {
        auto true_range = TileImpl::get_part_range(part);
        auto content_offset = TileImpl::get_part_offset(part);
        auto part_ac = tile[part].get_access<access::mode::read_write>();

        for (uindex_t c = 0; c < true_range.c; c++) {
            for (uindex_t r = 0; r < true_range.r; r++) {
                uindex_t word_i = (c * true_range.r + r) / word_length;
                uindex_t cell_i = (c * true_range.r + r) % word_length;
                part_ac[word_i][cell_i].value = ID(c + content_offset[0], r + content_offset[1]);
            }
        }
    }

    buffer<ID, 2> out_buffer(range<2>(tile_width, tile_height));
    tile.copy_to(out_buffer, id<2>(0, 0));

    auto out_buffer_ac = out_buffer.get_access<access::mode::read>();
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