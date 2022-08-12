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
#include "../Index.hpp"
#include <concepts>

namespace stencil {
namespace tdv {

template<typename V>
concept TimeDependentValue = std::copyable<V>;

template <typename F>
concept ValueFunction = requires(F function, uindex_t i_generation) {
    requires TimeDependentValue<typename F::Value>;
    { function(i_generation) } -> std::convertible_to<typename F::Value>;
};

template <typename T>
concept LocalState = TimeDependentValue<typename T::Value> && std::copyable<T> &&
    requires(T const &local_state, uindex_t i) {
    { local_state.get_value(i) } -> std::convertible_to<typename T::Value>;
};

template <typename T>
concept GlobalState = LocalState<typename T::LocalState> && std::copyable<T> &&
    requires(T const &global_state) {
    { global_state.prepare_local_state() } -> std::convertible_to<typename T::LocalState>;
};

template <typename T>
concept HostState = GlobalState<typename T::GlobalState> &&
    requires(T const &supplier, uindex_t i_generation, uindex_t n_generations) {
    {
        supplier.prepare_global_state(i_generation, n_generations)
        } -> std::convertible_to<typename T::GlobalState>;
};

} // namespace tdv
} // namespace stencil