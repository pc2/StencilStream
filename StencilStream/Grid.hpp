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
#include "Concepts.hpp"
#include "Index.hpp"
#include <CL/sycl.hpp>
#include <concepts>

namespace stencil {

template <typename G, typename Cell>
concept Grid = requires(G grid, cl::sycl::buffer<Cell, 2> buffer, uindex_t i) {
    { grid.copy_from_buffer(buffer) } -> std::same_as<void>;
    { grid.copy_to_buffer(buffer) } -> std::same_as<void>;
    { grid.get_grid_width() } -> std::convertible_to<uindex_t>;
    { grid.get_grid_height() } -> std::convertible_to<uindex_t>;
    { grid.get_i_generation() } -> std::convertible_to<uindex_t>;
    { grid.set_i_generation(i) } -> std::same_as<void>;
    { grid.inc_i_generation(i) } -> std::same_as<void>;
    { grid.make_similar() } -> std::same_as<G>;
};

} // namespace stencil