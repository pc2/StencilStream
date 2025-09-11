#pragma once

template <typename T> struct cell_members; // forward declaration

// 1) Impl: entfalte per index_sequence
template <typename TupleA, typename TupleB, typename F, std::size_t... Is>
void for_each_in_two_tuples_impl(TupleA &&a, TupleB &&b, F &&f, std::index_sequence<Is...>) {
    // für jedes Is ruf f(std::get<Is>(a), std::get<Is>(b), Is) auf
    (f(std::get<Is>(std::forward<TupleA>(a)), std::get<Is>(std::forward<TupleB>(b)), Is), ...);
}

// 2) Public helper: bestimmt die Länge aus TupleA (sollten gleich lang sein)
template <typename TupleA, typename TupleB, typename F>
void for_each_in_two_tuples(TupleA &&a, TupleB &&b, F &&f) {
    constexpr std::size_t N = std::tuple_size_v<std::decay_t<TupleA>>;
    static_assert(N == std::tuple_size_v<std::decay_t<TupleB>>, "Tuples must have same size");
    for_each_in_two_tuples_impl(std::forward<TupleA>(a), std::forward<TupleB>(b),
                                std::forward<F>(f), std::make_index_sequence<N>{});
}

template <typename CellT> auto make_buffers_from_member_ptrs(std::size_t N) {
    return std::apply(
        [N](auto... ptrs) {
            return std::make_tuple(
                sycl::buffer<std::remove_reference_t<decltype(std::declval<CellT>().*ptrs)>, 1>(
                    sycl::range<1>(N))...);
        },
        cell_members<CellT>::fields);
}

template <typename CellT>
using buffers_t_for = decltype(make_buffers_from_member_ptrs<CellT>(std::declval<std::size_t>()));
