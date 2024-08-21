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
#pragma once
#include <CL/sycl.hpp>
#include <boost/preprocessor/cat.hpp>
#include <cstdint>

namespace stencil {
#ifndef STENCIL_INDEX_WIDTH
    #define STENCIL_INDEX_WIDTH 64
#endif

/**
 * \brief An unsigned integer of configurable width.
 *
 * This integer type is used in StencilStream to indicate cell positions and iterations, among
 * others. Its width can be configured using the `STENCIL_INDEX_WIDTH` macro, which defaults to 64.
 *
 * Note that this type is likely to be replaced by `std::size_t` in a future update. Most
 * performance-critical indices already use custom-precision integers tailored to their specific
 * users, so the need to customize the width of the remaining indices is relatively low. In
 * addition, using `std::size_t` as a general index type would allow better compatibility with SYCL
 * and other libraries.
 */
using uindex_t = BOOST_PP_CAT(BOOST_PP_CAT(uint, STENCIL_INDEX_WIDTH), _t);

/**
 * \brief A signed integer of configurable width.
 *
 * This integer type is used in StencilStream to indicate cell positions and iterations, among
 * others. Its width can be configured using the `STENCIL_INDEX_WIDTH` macro, which defaults to 64.
 *
 * Note that this type is likely to be replaced by `std::ptrdiff_t` in a future update. Most
 * performance-critical indices already use custom-precision integers tailored to their specific
 * users, so the need to customize the width of the remaining indices is relatively low. In
 * addition, using `std::ptrdiff_t` as a general index type would allow better compatibility with
 * SYCL and other libraries.
 */
using index_t = BOOST_PP_CAT(BOOST_PP_CAT(int, STENCIL_INDEX_WIDTH), _t);
} // namespace stencil