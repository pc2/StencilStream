/*
 * Copyright © 2020-2026 Paderborn Center for Parallel Computing, Paderborn
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
    /**
     * \brief The type of values provided by the time-dependent value (TDV) system.
     *
     * This default definition uses `std::monostate`, which effectively disables the TDV feature:
     * the type carries no data and has exactly one value.
     */
    using TimeDependentValue = std::monostate;

    /**
     * \brief The radius of the stencil neighborhood.
     *
     * Defines how far the stencil extends from the central cell in each direction. A radius of 1
     * gives a 3×3 neighborhood; a radius of 2 gives a 5×5 neighborhood, and so on. This default
     * value of 1 can be overridden in a derived class.
     */
    static constexpr std::size_t stencil_radius = 1;

    /**
     * \brief The number of sub-iterations per iteration.
     *
     * Some stencil algorithms require multiple sub-passes per logical iteration (e.g., alternating
     * update schemes). This default value of 1 disables sub-iterations. It can be overridden in a
     * derived class.
     */
    static constexpr std::size_t n_subiterations = 1;

    /**
     * \brief Return the time-dependent value for the given iteration.
     *
     * This default implementation returns `std::monostate()`, which carries no information.
     * Override this method in a derived class to supply iteration-specific data to the transition
     * function.
     *
     * \param i_iteration The current (global) iteration index.
     * \return A `std::monostate` value.
     */
    std::monostate get_time_dependent_value(std::size_t i_iteration) const {
        return std::monostate();
    }
};

} // namespace stencil