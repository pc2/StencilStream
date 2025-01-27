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
#include <cmath>
#include <optional>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <unistd.h>

using namespace std;
using namespace sycl;

//////////////////////////////////////////////
// Needed physical constants for simulation //
//////////////////////////////////////////////
// velocity of light in m/s
constexpr float c0 = 299792458.0;

// Square root of two
constexpr float sqrt_2 = 1.4142135623730951;

constexpr float pi = 3.1415926535897932384626433;

constexpr size_t temporal_parallelism = 50;
constexpr size_t spatial_parallelism = 2;

/* stencil parameters */
constexpr size_t tile_width = 512;

#if defined(STENCILSTREAM_BACKEND_TILING)
// tiling, make tile as wide as possible.
constexpr size_t tile_height = 1 << 16;
#else
// monotile and CPU. More than a quadratic tile doesn't make sense.
constexpr size_t tile_height = tile_width;
#endif

constexpr size_t max_n_rings = 15;
constexpr size_t bits_max_n_rings = std::bit_width(max_n_rings + 1);
using uindex_ring_t = ac_int<bits_max_n_rings, false>;