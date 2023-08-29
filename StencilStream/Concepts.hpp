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

namespace stencil {

template <typename T>
concept TransitionFunction =
    requires {
        // cell type
        requires std::semiregular<typename T::Cell>;
    } &&
    requires {
        // stencil radius
        requires std::same_as<decltype(T::stencil_radius), const uindex_t>;
        requires(T::stencil_radius >= 1);
    } &&
    requires {
        // number of subgenerations
        requires std::same_as<decltype(T::n_subgenerations), const uindex_t>;
    } &&
    requires {
        // time-dependent value
        requires std::copyable<typename T::TimeDependentValue>;
    } &&
    requires(T trans_func,
             Stencil<typename T::Cell, T::stencil_radius, typename T::TimeDependentValue> const
                 &stencil) {
        // update method
        { trans_func(stencil) } -> std::convertible_to<typename T::Cell>;
    };

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
    requires(T &supplier, cl::sycl::handler &cgh, uindex_t i_generation, uindex_t n_generations) {
        {
            // building the global state
            supplier.build_kernel_argument(cgh, i_generation, n_generations)
        } -> std::convertible_to<typename T::KernelArgument>;
    } && requires(T &supplier, uindex_t i_generation, uindex_t n_generations) {
        // preparing the global state
        { supplier.prepare_range(i_generation, n_generations) };
    };

} // namespace tdv
} // namespace stencil