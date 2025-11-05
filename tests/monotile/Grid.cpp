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
#include <StencilStream/monotile/Grid.hpp>

using namespace stencil;
using namespace stencil::monotile;
using namespace sycl;
using namespace std;

// Assert that the monotile grid fulfills the grid concept.
static_assert(concepts::Grid<Grid<sycl::id<2>>, sycl::id<2>>);

constexpr std::size_t max_grid_height = 64;
constexpr std::size_t max_grid_width = 32;

TEST_CASE("monotile::Grid::Grid", "[monotile::Grid]") {
    grid_test::test_constructors<Grid<sycl::id<2>>>(max_grid_height, max_grid_width);
}

TEST_CASE("monotile::Grid::copy_from_buffer", "[monotile::Grid]") {
    grid_test::test_copy_from_buffer<Grid<sycl::id<2>>>(max_grid_height, max_grid_width);
}

TEST_CASE("monotile::Grid::copy_to_buffer", "[monotile::Grid]") {
    grid_test::test_copy_to_buffer<Grid<sycl::id<2>>>(max_grid_height, max_grid_width);
}

TEST_CASE("monotile::Grid::make_similar", "[monotile::Grid]") {
    grid_test::test_make_similar<Grid<sycl::id<2>>>(max_grid_height, max_grid_width);
}
