/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "../Concepts.hpp"

namespace stencil {
namespace tdv {
namespace single_pass {

template <typename T, typename TransFunc>
concept LocalState = stencil::concepts::TransitionFunction<TransFunc> &&
                     requires(T const &local_state, uindex_t i) {
                         {
                             local_state.get_time_dependent_value(i)
                             } -> std::same_as<typename TransFunc::TimeDependentValue>;
                     };

template <typename T, typename TransFunc>
concept KernelArgument = stencil::concepts::TransitionFunction<TransFunc> &&
                         LocalState<typename T::LocalState, TransFunc> && std::copyable<T> &&
                         std::constructible_from<typename T::LocalState, T const &>;

template <typename T, typename TransFunc>
concept GlobalState =
    stencil::concepts::TransitionFunction<TransFunc> &&
    std::constructible_from<T, TransFunc, uindex_t, uindex_t> &&
    KernelArgument<typename T::KernelArgument, TransFunc> &&
    std::constructible_from<typename T::KernelArgument, T &, sycl::handler &, uindex_t, uindex_t>;

template <typename T, typename TransFunc, uindex_t max_n_generations>
concept Strategy =
    stencil::concepts::TransitionFunction<TransFunc> &&
    GlobalState<typename T::template GlobalState<TransFunc, max_n_generations>, TransFunc>;

struct InlineStrategy {
    template <stencil::concepts::TransitionFunction TransFunc, uindex_t max_n_generations>
    struct GlobalState {
        using TDV = typename TransFunc::TimeDependentValue;

        GlobalState(TransFunc trans_func, uindex_t generation_offset, uindex_t n_generations)
            : trans_func(trans_func) {}

        struct KernelArgument {
            KernelArgument(GlobalState &global_state, sycl::handler &cgh,
                           uindex_t generation_offset, uindex_t n_generations)
                : trans_func(global_state.trans_func), generation_offset(generation_offset) {}

            using LocalState = KernelArgument;

            TDV get_time_dependent_value(uindex_t i_generation) const {
                return trans_func.get_time_dependent_value(generation_offset + i_generation);
            }

          private:
            TransFunc trans_func;
            uindex_t generation_offset;
        };

      private:
        TransFunc trans_func;
    };
};

struct PrecomputeOnDeviceStrategy {
    template <stencil::concepts::TransitionFunction TransFunc, uindex_t max_n_generations>
    struct GlobalState {
        using TDV = typename TransFunc::TimeDependentValue;

        GlobalState(TransFunc trans_func, uindex_t generation_offset, uindex_t n_generations)
            : trans_func(trans_func) {}

        struct KernelArgument {
            KernelArgument(GlobalState &global_state, sycl::handler &cgh,
                           uindex_t generation_offset, uindex_t n_generations)
                : trans_func(global_state.trans_func), generation_offset(generation_offset) {}

            struct LocalState {
                LocalState(KernelArgument const &kernel_argument) : values() {
                    for (uindex_t i = 0; i < max_n_generations; i++) {
                        values[i] = kernel_argument.trans_func.get_time_dependent_value(
                            kernel_argument.generation_offset + i);
                    }
                }

                TDV get_time_dependent_value(uindex_t i_generation) const {
                    return values[i_generation];
                }

              private:
                TDV values[max_n_generations];
            };

          private:
            TransFunc trans_func;
            uindex_t generation_offset;
        };

      private:
        TransFunc trans_func;
    };
};

struct PrecomputeOnHostStrategy {
    template <stencil::concepts::TransitionFunction TransFunc, uindex_t max_n_generations>
    class GlobalState {
      public:
        using TDV = typename TransFunc::TimeDependentValue;

        GlobalState(TransFunc function, uindex_t generation_offset, uindex_t n_generations)
            : function(function), generation_offset(generation_offset),
              value_buffer(sycl::range<1>(n_generations)) {
            sycl::host_accessor ac(value_buffer, sycl::read_write);
            for (uindex_t i = 0; i < n_generations; i++) {
                ac[i] = function.get_time_dependent_value(generation_offset + i);
            }
        }

        struct KernelArgument {
            KernelArgument(GlobalState &global_state, sycl::handler &cgh, uindex_t i_generation,
                           uindex_t n_generations)
                : ac() {
                assert(n_generations <= max_n_generations);
                assert(i_generation >= global_state.generation_offset);
                assert(i_generation + n_generations <=
                       global_state.generation_offset + global_state.value_buffer.get_range()[0]);

                sycl::range<1> access_range(n_generations);
                sycl::id<1> access_offset(i_generation - global_state.generation_offset);
                ac = sycl::accessor<TDV, 1, sycl::access::mode::read>(
                    global_state.value_buffer, cgh, access_range, access_offset);
            }

            struct LocalState {
                LocalState(KernelArgument const &kernel_argument) : values() {
                    uindex_t n_values =
                        std::min(max_n_generations, uindex_t(kernel_argument.ac.get_range()[0]));

                    for (uindex_t i = 0; i < n_values; i++)
                        values[i] = kernel_argument.ac[i];
                }

                TDV get_time_dependent_value(uindex_t i) const { return values[i]; }

              private:
                TDV values[max_n_generations];
            };

          private:
            sycl::accessor<TDV, 1, sycl::access::mode::read> ac;
        };

      private:
        TransFunc function;
        uindex_t generation_offset;
        sycl::buffer<TDV, 1> value_buffer;
    };
};

} // namespace single_pass
} // namespace tdv
} // namespace stencil