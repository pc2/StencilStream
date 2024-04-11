/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
 * University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once
#include <StencilStream/Helpers.hpp>
#include <StencilStream/Index.hpp>

const stencil::uindex_t tile_width = 64;
const stencil::uindex_t tile_height = 32;

const stencil::uindex_t subiterations = 2;
const stencil::uindex_t n_processing_elements = 4;
const stencil::uindex_t iters_per_pass = n_processing_elements / subiterations;
const stencil::uindex_t stencil_radius = 2;
const stencil::uindex_t halo_radius = n_processing_elements * stencil_radius;
const stencil::uindex_t core_width = tile_width - 2 * halo_radius;
const stencil::uindex_t core_height = tile_height - 2 * halo_radius;

const stencil::uindex_t grid_width = 128;
const stencil::uindex_t grid_height = 64;