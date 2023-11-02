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
#include "Index.hpp"
#include "Stencil.hpp"

#include <concepts>
#include <type_traits>

namespace stencil {
namespace concepts {
template <typename T>
concept TransitionFunction =
    // Cells must to be semiregular.
    std::semiregular<typename T::Cell> &&
    // Time-dependent values must be copyable.
    std::copyable<typename T::TimeDependentValue> &&
    // The stencil radius must be a constant greater than 1.
    std::same_as<decltype(T::stencil_radius), const uindex_t> && (T::stencil_radius >= 1) &&
    // The number of subgenerations must be a constant greater than 1.
    std::same_as<decltype(T::n_subgenerations), const uindex_t> && (T::n_subgenerations >= 1) &&
    // The transition function must be invocable. Its argument must be a stencil and its return
    // value must be a cell.
    requires(T trans_func,
             Stencil<typename T::Cell, T::stencil_radius, typename T::TimeDependentValue> const
                 &stencil) {
        // update method
        { trans_func(stencil) } -> std::same_as<typename T::Cell>;
    };

template <typename Accessor, typename Cell>
concept GridAccessor = requires(Accessor ac, uindex_t c, uindex_t r) {
    { ac[sycl::id<2>(c, r)] } -> std::same_as<Cell &>;
    { ac[c][r] } -> std::same_as<Cell &>;
};

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

template <typename SU, typename TF, typename TDVH, typename G>
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
        { params.generation_offset } -> std::same_as<uindex_t &>;
        { params.n_generations } -> std::same_as<uindex_t &>;
        { params.tdv_host_state } -> std::same_as<TDVH &>;
        { params.device } -> std::same_as<sycl::device &>;
        { params.blocking } -> std::same_as<bool &>;
    } && TransitionFunction<TF> && Grid<G, typename TF::Cell> &&
    (std::is_class<typename SU::Params>::value);

namespace tdv {

template <typename F>
concept ValueFunction = requires(F const &function, uindex_t i_generation) {
    requires std::copyable<typename F::Value>;
    { function(i_generation) } -> std::convertible_to<typename F::Value>;
};

template <typename T>
concept LocalState = std::copyable<typename T::Value> && std::copyable<T> &&
                     requires(T const &local_state, uindex_t i) {
                         { local_state.get_value(i) } -> std::convertible_to<typename T::Value>;
                     };

template <typename T>
concept KernelArgument =
    LocalState<typename T::LocalState> && std::copyable<T> && requires(T const &global_state) {
        { global_state.build_local_state() } -> std::convertible_to<typename T::LocalState>;
    };

template <typename T>
concept HostState =
    KernelArgument<typename T::KernelArgument> &&
    requires(T &supplier, sycl::handler &cgh, uindex_t i_generation, uindex_t n_generations) {
        {
            // building the global state
            supplier.build_kernel_argument(cgh, i_generation, n_generations)
        } -> std::convertible_to<typename T::KernelArgument>;
    } && requires(T &supplier, uindex_t i_generation, uindex_t n_generations) {
        // preparing the global state
        { supplier.prepare_range(i_generation, n_generations) };
    };

} // namespace tdv
} // namespace concepts
} // namespace stencil