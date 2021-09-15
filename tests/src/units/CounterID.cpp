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
#include <StencilStream/CounterID.hpp>
#include <res/catch.hpp>
#include <res/constants.hpp>

using namespace std;
using namespace stencil;

using CID = CounterID<uindex_t>;

TEST_CASE("CounterID::operator++", "[CounterID]") {
    CID counter(0, 0, tile_width, tile_height);
    for (uindex_t c = 0; c < tile_width; c++) {
        for (uindex_t r = 0; r < tile_height; r++) {
            REQUIRE(counter.c == c);
            REQUIRE(counter.r == r);
            counter++;
        }
    }
    REQUIRE(counter.c == 0);
    REQUIRE(counter.r == 0);
}

TEST_CASE("CounterID::operator--", "[CounterID]") {
    CID counter(tile_width - 1, tile_height - 1, tile_width, tile_height);
    for (index_t c = tile_width - 1; c >= 0; c--) {
        for (index_t r = tile_height - 1; r >= 0; r--) {
            REQUIRE(counter.c == c);
            REQUIRE(counter.r == r);
            counter--;
        }
    }
    REQUIRE(counter.c == tile_width - 1);
    REQUIRE(counter.r == tile_height - 1);
}