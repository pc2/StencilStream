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
#include <sycl/sycl.hpp>
#include <stdint.h>
#include <type_traits>
#include <concepts>

namespace stencil {
namespace cuda {
namespace internal {

// Create runtime tuple of sycl::buffer<T,1> for each member-pointer in cell_members<CellT>::fields.
// The element type for each buffer is deduced from the member-pointer ptrs via:
// decltype(std::declval<CellT>().*ptrs)
// std::remove_reference_t<> strips references to get the raw field type.
template <typename CellT> auto alloc_field_buffers(std::size_t N) {
    return std::apply(
        [N](auto... ptrs) {
            return std::make_tuple(
                sycl::buffer<std::remove_reference_t<decltype(std::declval<CellT>().*ptrs)>, 1>(
                    sycl::range<1>(N))...);
        },
        CellT::fields);
}

// FieldBuffers<CellT> is the tuple type returned by alloc_field_buffers<CellT>(size_t).
template <typename CellT>
using FieldBuffers = decltype(alloc_field_buffers<CellT>(std::declval<std::size_t>()));

// Expand two tuples by using an index sequence.
template <typename TupleA, typename TupleB, typename F, std::size_t... Is>
void for_each_in_two_tuples_impl(TupleA &&a, TupleB &&b, F &&f, std::index_sequence<Is...>) {
    // The fold-expression expands to multiple calls, one per tuple element.
    (f(std::get<Is>(std::forward<TupleA>(a)), std::get<Is>(std::forward<TupleB>(b))), ...);
}

// - TupleA: tuple of accessors
// - TupleB: tuple of member-pointers
// - F: Function invoked as f(element_from_a, element_from_b)
template <typename TupleA, typename TupleB, typename F>
void for_each_in_two_tuples(TupleA &&a, TupleB &&b, F &&f) {
    constexpr std::size_t N = std::tuple_size_v<std::decay_t<TupleA>>;
    static_assert(N == std::tuple_size_v<std::decay_t<TupleB>>, "Tuples must have same size");
    for_each_in_two_tuples_impl(std::forward<TupleA>(a), std::forward<TupleB>(b),
                                std::forward<F>(f), std::make_index_sequence<N>{});
}

}
}
}