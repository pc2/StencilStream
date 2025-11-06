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
#include "../GridTest_cuda_soa.hpp"
#include <StencilStream/cuda-soa/Grid.hpp>

using namespace stencil;
using namespace stencil::cuda;

struct TestCell {
    sycl::id<2> id;
};

template <> struct cell_members<TestCell> {
    static constexpr auto fields = std::make_tuple(&TestCell::id);
};

using TestGrid = Grid<TestCell>;

static_assert(concepts::Grid<TestGrid, TestCell>);

TEST_CASE("cuda-soa::Grid::Grid", "[cuda-soa::Grid]") {
    grid_test::test_constructors<TestCell, TestGrid>(128, 128);
}

TEST_CASE("cuda-soa::Grid::copy_from_buffer", "[cuda-soa::Grid]") {
    grid_test::test_copy_from_buffer<TestCell, TestGrid>(128, 128);
}

TEST_CASE("cuda-soa::Grid::copy_to_buffer", "[cuda-soa::Grid]") {
    grid_test::test_copy_to_buffer<TestCell, TestGrid>(128, 128);
}

TEST_CASE("cuda-soa::Grid::make_similar", "[cuda-soa::Grid]") {
    grid_test::test_make_similar<TestCell, TestGrid>(128, 128);
}
