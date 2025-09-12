#pragma once
#include <cstddef>
#include <utility>

template <typename T> struct cell_members;

// Expand two tuples in using an index sequence.
template <typename TupleA, typename TupleB, typename F, std::size_t... Is>
void for_each_in_two_tuples_impl(TupleA &&a, TupleB &&b, F &&f, std::index_sequence<Is...>) {
    // The fold-expression expands to multiple calls, one per tuple element.
    (f(std::get<Is>(std::forward<TupleA>(a)), std::get<Is>(std::forward<TupleB>(b))), ...);
}

// - TupleA: tuple of accessors (or destination objects)
// - TupleB: tuple of member-pointers (or source descriptors)
// - F: Function invoked as f(element_from_a, element_from_b)
template <typename TupleA, typename TupleB, typename F>
void for_each_in_two_tuples(TupleA &&a, TupleB &&b, F &&f) {
    constexpr std::size_t N = std::tuple_size_v<std::decay_t<TupleA>>;
    static_assert(N == std::tuple_size_v<std::decay_t<TupleB>>, "Tuples must have same size");
    for_each_in_two_tuples_impl(std::forward<TupleA>(a), std::forward<TupleB>(b),
                                std::forward<F>(f), std::make_index_sequence<N>{});
}

// Create runtime tuple of sycl::buffer<T,1> for each member-pointer in cell_members<CellT>::fields.
// The element type for each buffer is deduced from the member-pointer ptrs via:
// decltype(std::declval<CellT>().*ptrs)
// std::remove_reference_t<> strips references to get the raw field type.
template <typename CellT> auto make_buffers_from_member_ptrs(std::size_t N) {
    return std::apply(
        [N](auto... ptrs) {
            return std::make_tuple(
                sycl::buffer<std::remove_reference_t<decltype(std::declval<CellT>().*ptrs)>, 1>(
                    sycl::range<1>(N))...);
        },
        cell_members<CellT>::fields);
}

// buffers_t_for<CellT> is the tuple type returned by make_buffers_from_member_ptrs<CellT>(size_t).
template <typename CellT>
using buffers_t_for = decltype(make_buffers_from_member_ptrs<CellT>(std::declval<std::size_t>()));