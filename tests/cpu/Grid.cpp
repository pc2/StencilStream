/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "../GridTest.hpp"
#include "../constants.hpp"
#include <StencilStream/cpu/Grid.hpp>

using namespace stencil;
using namespace stencil::cpu;

using TestGrid = Grid<ID>;

static_assert(concepts::Grid<TestGrid, ID>);

TEST_CASE("cpu::Grid::Grid", "[cpu::Grid]") {
    grid_test::test_constructors<TestGrid>(tile_width, tile_height);
}

TEST_CASE("cpu::Grid::copy_from_buffer", "[cpu::Grid]") {
    grid_test::test_copy_from_buffer<TestGrid>(tile_width, tile_height);
}

TEST_CASE("cpu::Grid::copy_to_buffer", "[cpu::Grid]") {
    grid_test::test_copy_to_buffer<TestGrid>(tile_width, tile_height);
}

TEST_CASE("cpu::Grid::make_similar", "[cpu::Grid]") {
    grid_test::test_make_similar<TestGrid>(tile_width, tile_height);
}
