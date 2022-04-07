/*
 * Copyright © 2020-2022 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include <StencilStream/Index.hpp>
#include <cmath>
#include <optional>
#include <unistd.h>

using namespace std;
using namespace cl::sycl;
using namespace stencil;

//////////////////////////////////////////////
// Needed physical constants for simulation //
//////////////////////////////////////////////
// velocity of light in m/s
constexpr float c0 = 299792458.0;

// Square root of two
constexpr float sqrt_2 = 1.4142135623730951;

constexpr float pi = 3.1415926535897932384626433;

/* stencil parameters */
constexpr uindex_t tile_height = 512;
constexpr uindex_t tile_width = tile_height;

constexpr uindex_t stencil_radius = 1;

#if MATERIAL == 0 && SOURCE == 1 && EXECUTOR == 0
// material coefficients in cells, source LUT, monotile
constexpr uindex_t pipeline_length = 320;

#elif MATERIAL == 0 && SOURCE == 1 && EXECUTOR == 1
// material coefficients in cells, source LUT, tiling
constexpr uindex_t pipeline_length = 186;

#else
// fallback
constexpr uindex_t pipeline_length = 100;

#endif

static_assert(pipeline_length % 2 == 0);
