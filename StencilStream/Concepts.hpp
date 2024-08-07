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
#include "Index.hpp"
#include "Stencil.hpp"

#include <concepts>
#include <type_traits>

namespace stencil {
namespace concepts {

/**
 * \brief A technical definition of a stencil transition function.
 * 
 * This concept lists all required type definitions, constants, and methods that a \ref stencil::concepts::StencilUpdate "StencilUpdate" needs from a transition function.
 * This includes multiple definitions for advanced features.
 * Most users are adviced to extend the \ref stencil::BaseTransitionFunction class.
 * It provides default definitions that disable those features.
 * 
 * The required type definitions are:
 * * `Cell`: The cell type the transition function operates on. It must be semiregular, i.e. copyable bit by bit.
 * * `TimeDependentValue`: The type of the time-dependent value computed by the `get_time_dependent_value` method. It must also be semiregular.
 * 
 * The required constants are:
 * * `uindex_t stencil_radius`: The radius of the stencil. It must be greater than or equal to 1.
 * * `uindex_t n_subiterations`: The number of sub-iterations of the transition function. It must be greater than or equal to 1.
 * 
 * The required methods are:
 * * `Cell operator()(Stencil<Cell, stencil_radius> const&stencil) const`: Compute the next iteration of the stencil's central cell. This method must be pure, i.e. it must not modify either the stencil's or the transition function's state.
 * * `TimeDependentValue get_time_dependent_value(uindex_t i_iteration) const`: Compute the time-dependent value for the given iteration. This method must be pure, i.e. it must not modify the transition function's state.
 */
template <typename T>
concept TransitionFunction =
    std::semiregular<typename T::Cell> &&
    std::copyable<typename T::TimeDependentValue> &&
    
    std::same_as<decltype(T::stencil_radius), const uindex_t> && (T::stencil_radius >= 1) &&
    std::same_as<decltype(T::n_subiterations), const uindex_t> && (T::n_subiterations >= 1) &&
    
    requires(T const &trans_func,
             Stencil<typename T::Cell, T::stencil_radius, typename T::TimeDependentValue> const
                 &stencil) {
        { trans_func(stencil) } -> std::same_as<typename T::Cell>;
    } &&
    requires(T const &trans_func, uindex_t i_iteration) {
        {
            trans_func.get_time_dependent_value(i_iteration)
        } -> std::same_as<typename T::TimeDependentValue>;
    };

/**
 * \brief An accessor for a two-dimensional grid.
 * 
 * It must provide access either via a `sycl::id<2>` object or via two consecutive accesses with `uindex_t`s.
 */
template <typename Accessor, typename Cell>
concept GridAccessor = requires(Accessor ac, uindex_t c, uindex_t r) {
    { ac[sycl::id<2>(c, r)] } -> std::same_as<Cell &>;
    { ac[c][r] } -> std::same_as<Cell &>;
};

/**
 * \brief A container for cells.
 * 
 * This concept defines certain methods and properties that all grids should share, so that backend-independent applications can be written.
 * 
 * First of all, each grid must contain a template class called `GridAccessor` that, given an instance of `sycl::access::mode`, fulfills the \ref stencil::concepts::GridAccessor "GridAccessor" concept.
 */
template <typename G, typename Cell>
concept Grid = requires(G &grid, sycl::buffer<Cell, 2> buffer, uindex_t c, uindex_t r, Cell cell) {
    { G(c, r) } -> std::same_as<G>;
    { G(sycl::range<2>(c, r)) } -> std::same_as<G>;
    { G(buffer) } -> std::same_as<G>;
    { grid.copy_from_buffer(buffer) } -> std::same_as<void>;
    { grid.copy_to_buffer(buffer) } -> std::same_as<void>;
    { grid.get_grid_width() } -> std::convertible_to<uindex_t>;
    { grid.get_grid_height() } -> std::convertible_to<uindex_t>;
    { grid.make_similar() } -> std::same_as<G>;
    {
        typename G::template GridAccessor<sycl::access::mode::read_write>(grid)
    } -> GridAccessor<Cell>;
};

template <typename SU, typename TF, typename G>
concept StencilUpdate =
    // Test construction and update call.
    requires(SU stencil_update, G &grid, typename SU::Params params) {
        { SU(params) } -> std::same_as<SU>;
        { stencil_update.get_params() } -> std::same_as<typename SU::Params &>;
        { stencil_update(grid) } -> std::same_as<G>;
    } &&
    // Test existance of parameter fields
    requires(typename SU::Params params) {
        { params.transition_function } -> std::same_as<TF &>;
        { params.halo_value } -> std::same_as<typename TF::Cell &>;
        { params.iteration_offset } -> std::same_as<uindex_t &>;
        { params.n_iterations } -> std::same_as<uindex_t &>;
        { params.device } -> std::same_as<sycl::device &>;
    } && TransitionFunction<TF> && Grid<G, typename TF::Cell> &&
    (std::is_class<typename SU::Params>::value);

} // namespace concepts
} // namespace stencil