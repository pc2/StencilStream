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
#pragma once
#include <CL/sycl.hpp>
#include <boost/preprocessor/cat.hpp>
#include <cstdint>

namespace stencil {
#ifndef STENCIL_INDEX_WIDTH
    #define STENCIL_INDEX_WIDTH 64
#endif

/**
 * Integer types for indexing.
 *
 * There are different index types available for different contexts, with different widths. Their
 * general schema is `[u]index_(type_)t`, where the optional leading `u` denotes whether the index
 * is signed, and where `type` denotes the type or context of the index.
 *
 * The different types are tailored towards different contexts and therefore have different widths.
 * The `min` indices are used in the context of stencil buffers, i.e. to index a cell in a stencil
 * buffer. The `1d` indices are used in the context of one physical dimension, i.e. to index the row
 * or column of a cell in a tile buffer. The `2d` indices are used in the context of combined tile
 * row and column indices, i.e. when iterating over a tile with a single coalesced index. Lastly,
 * the type can also be omited, which yields the widest index possible, which is used for example to
 * denote a generation index or a cell in the grid.
 *
 * The exact widths are set using the `STENCIL_INDEX_(TYPE_)WIDTH` macros. Their dimensions are
 * reasonable for FPGAs in early 2022, but can be changed as necesssary. Static asserts throughout
 * the library ensure that the used index type is wide enough to hold certain values. Therefore, you
 * can in theory decrease the until you get compilation errors.
 */
typedef BOOST_PP_CAT(BOOST_PP_CAT(uint, STENCIL_INDEX_WIDTH), _t) uindex_t;
typedef BOOST_PP_CAT(BOOST_PP_CAT(int, STENCIL_INDEX_WIDTH), _t) index_t;
} // namespace stencil