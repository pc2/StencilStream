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
#pragma once
#include <boost/preprocessor/cat.hpp>
#include <cstdint>

namespace stencil {
#ifndef STENCIL_INDEX_WIDTH
    #define STENCIL_INDEX_WIDTH 64
#endif

/**
 * Integer types for indexing.
 *
 * There is always a signed version, `index_t`, and an unsigned version, `uindex_t`. Their width is
 * defined by the `STENCIL_INDEX_WIDTH` macro. The default is 64 and can be increased to allow
 * bigger buffers or decreased to reduce the complexity and resource requirements.
 *
 * Static asserts throughout the library ensure that the index type is wide enough. Therefore, you
 * can decrease the until you get compilation errors.
 */
typedef BOOST_PP_CAT(BOOST_PP_CAT(uint, STENCIL_INDEX_WIDTH), _t) uindex_t;
typedef BOOST_PP_CAT(BOOST_PP_CAT(int, STENCIL_INDEX_WIDTH), _t) index_t;
} // namespace stencil