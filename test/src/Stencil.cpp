/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "catch.hpp"
#include <Stencil.hpp>

using namespace stencil;

const Index radius = 2;

TEST_CASE("The diameter is correct", "[Stencil]") {
    Stencil<Index, 2> stencil;

    REQUIRE(stencil.diameter() == Stencil<Index, 2>::diameter());
    REQUIRE(stencil.diameter() == 2*radius + 1);
};

TEST_CASE("Signed writing and unsigned reading are correct", "[Stencil]") {
    Stencil<Index, 2> stencil;

    for (Index c = -radius; c <= radius; c++)
    {
        for (Index r = -radius; r <= radius; r++)
        {
            stencil[ID(c, r)] = c + r;
        }
    }

    for (UIndex c = 0; c < stencil.diameter(); c++)
    {
        for (UIndex r = 0; r < stencil.diameter(); r++)
        {
            REQUIRE(stencil[UID(c, r)] == Index(c) + Index(r) - 2*radius);
        }
    }
};

TEST_CASE("Unsigned writing and signed reading are correct", "[Stencil]") {
    Stencil<Index, 2> stencil;

    for (UIndex c = 0; c < stencil.diameter(); c++)
    {
        for (UIndex r = 0; r < stencil.diameter(); r++)
        {
            stencil[UID(c, r)] = c + r;
        }
    }

    for (Index c = -radius; c <= radius; c++)
    {
        for (Index r = -radius; r <= radius; r++)
        {
            REQUIRE(stencil[ID(c, r)] == Index(c) + Index(r) + 2*radius);
        }
    }
};