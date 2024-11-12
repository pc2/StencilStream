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
#include "constants.hpp"
#include <StencilStream/Stencil.hpp>
#include <catch2/catch_all.hpp>

using namespace stencil;

using StencilImpl = Stencil<int, 2>;

TEST_CASE("Stencil::diameter", "[Stencil]") {
    StencilImpl stencil(sycl::id<2>(0, 0), sycl::range<2>(42, 42), 0, 0, std::monostate());

    REQUIRE(stencil.diameter == StencilImpl::diameter);
    REQUIRE(stencil.diameter == 2 * stencil_radius + 1);
};

TEST_CASE("Stencil::operator[](int)", "[Stencil]") {
    StencilImpl stencil(sycl::id<2>(0, 0), sycl::range<2>(42, 42), 0, 0, std::monostate());

    for (std::size_t r = 0; r < stencil.diameter; r++) {
        for (std::size_t c = 0; c < stencil.diameter; c++) {
            stencil[sycl::id<2>(r, c)] = int(r) + int(c) - 2 * stencil_radius;
        }
    }

    for (int r = -int(stencil_radius); r <= int(stencil_radius); r++) {
        for (int c = -int(stencil_radius); c <= int(stencil_radius); c++) {
            REQUIRE(stencil[r][c] == r + c);
        }
    }
};