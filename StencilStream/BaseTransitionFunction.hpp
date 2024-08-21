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
#include "Concepts.hpp"
#include <variant>

namespace stencil {

/**
 * \brief Base class for transition functions that disables advanced features.
 *
 * Users who want to start implementing a \ref stencil::concepts::TransitionFunction
 * "transition function" should let it inherit this class. It provides default definitions that
 * disable certain advanced StencilStream features, for example the time-dependent value feature or
 * sub-iterations.
 *
 * For the time-dependent value system, this class uses the `std::monostate` type. This type has
 * only one value, which is "computed" for every iteration.
 */
class BaseTransitionFunction {
  public:
    using TimeDependentValue = std::monostate;

    static constexpr uindex_t stencil_radius = 1;
    static constexpr uindex_t n_subiterations = 1;

    std::monostate get_time_dependent_value(uindex_t i_iteration) const { return std::monostate(); }
};

} // namespace stencil