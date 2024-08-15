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

using StencilImpl = Stencil<index_t, 2>;
using StencilID = typename StencilImpl::StencilID;
using StencilUID = typename StencilImpl::StencilUID;

TEST_CASE("Stencil::diameter", "[Stencil]") {
    StencilImpl stencil(ID(0, 0), UID(42, 42), 0, 0, std::monostate());

    REQUIRE(stencil.diameter == StencilImpl::diameter);
    REQUIRE(stencil.diameter == 2 * stencil_radius + 1);
};

TEST_CASE("Stencil::operator[](ID)", "[Stencil]") {
    StencilImpl stencil(ID(0, 0), UID(42, 42), 0, 0, std::monostate());

    for (index_t c = -stencil_radius; c <= index_t(stencil_radius); c++) {
        for (index_t r = -stencil_radius; r <= index_t(stencil_radius); r++) {
            stencil[ID(c, r)] = c + r;
        }
    }

    for (uindex_t c = 0; c < stencil.diameter; c++) {
        for (uindex_t r = 0; r < stencil.diameter; r++) {
            REQUIRE(stencil[UID(c, r)] == index_t(c) + index_t(r) - 2 * stencil_radius);
        }
    }
};

TEST_CASE("Stencil::operator[](UID)", "[Stencil]") {
    StencilImpl stencil(ID(0, 0), UID(42, 42), 0, 0, std::monostate());

    for (uindex_t c = 0; c < stencil.diameter; c++) {
        for (uindex_t r = 0; r < stencil.diameter; r++) {
            stencil[UID(c, r)] = c + r;
        }
    }

    for (index_t c = -stencil_radius; c <= stencil_radius; c++) {
        for (index_t r = -stencil_radius; r <= stencil_radius; r++) {
            REQUIRE(stencil[ID(c, r)] == index_t(c) + index_t(r) + 2 * stencil_radius);
        }
    }
};

TEST_CASE("Stencil::operator[](StencilID)", "[Stencil]") {
    StencilImpl stencil(ID(0, 0), UID(42, 42), 0, 0, std::monostate());

    for (index_t c = -stencil_radius; c <= index_t(stencil_radius); c++) {
        for (index_t r = -stencil_radius; r <= index_t(stencil_radius); r++) {
            stencil[StencilID(c, r)] = c + r;
        }
    }

    for (uindex_t c = 0; c < stencil.diameter; c++) {
        for (uindex_t r = 0; r < stencil.diameter; r++) {
            REQUIRE(stencil[StencilUID(c, r)] == index_t(c) + index_t(r) - 2 * stencil_radius);
        }
    }
};

TEST_CASE("Stencil::operator[](StencilUID)", "[Stencil]") {
    StencilImpl stencil(ID(0, 0), UID(42, 42), 0, 0, std::monostate());

    for (uindex_t c = 0; c < stencil.diameter; c++) {
        for (uindex_t r = 0; r < stencil.diameter; r++) {
            stencil[StencilUID(c, r)] = c + r;
        }
    }

    for (index_t c = -stencil_radius; c <= stencil_radius; c++) {
        for (index_t r = -stencil_radius; r <= stencil_radius; r++) {
            REQUIRE(stencil[StencilID(c, r)] == index_t(c) + index_t(r) + 2 * stencil_radius);
        }
    }
};