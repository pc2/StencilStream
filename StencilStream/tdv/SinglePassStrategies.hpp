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

/**
 * \brief Implementations of the TDV system for single-pass backends.
 *
 * These include the tiling and monotile FPGA backends as well as the CPU backend.
 */
namespace single_pass {

/**
 * \brief The requirements for a pass-local TDV system state.
 *
 * Each local state is instantiated just prior to starting a pass, and the final time-dependent
 * values are extracted from it.
 *
 * The required methods are:
 *
 * * `TDV get_time_dependent_value(uindex_t i) const`: Return the time-dependent value for the
 * pass-local iteration `i`. This is the current iteration minus the iteration index offset of the
 * current pass. The type `TDV` has to be the same as `TransFunc::TimeDependentValue`.
 *
 * \tparam TransFunc The transition function that contains the TDV definition.
 */
template <typename T, typename TransFunc>
concept LocalState =
    stencil::concepts::TransitionFunction<TransFunc> && requires(T const &local_state, uindex_t i) {
        {
            local_state.get_time_dependent_value(i)
        } -> std::same_as<typename TransFunc::TimeDependentValue>;
    };

/**
 * \brief The requirements for a TDV kernel argument.
 *
 * Each kernel argument is constructed on the host and then passed to the execution kernel. From
 * this kernel argument, the execution constructs the \ref stencil::tdv::single_pass::LocalState
 * "LocalState".
 *
 * \tparam TransFunc The transition function that contains the TDV definition.
 */
template <typename T, typename TransFunc>
concept KernelArgument = stencil::concepts::TransitionFunction<TransFunc> &&
                         LocalState<typename T::LocalState, TransFunc> && std::copyable<T> &&
                         std::constructible_from<typename T::LocalState, T const &>;

/**
 * \brief The requirements for a TDV system's global state.
 *
 * This global state is constructed and stored on the host. It is constructed by the \ref
 * stencil::concepts::StencilUpdate "StencilUpdate" from the transition function as well as the
 * iteration index offset and the number of iterations that are requested from the user. For
 * example, if we have a transition function object `tf` and the user has requested to compute the
 * iterations 17 to 42, the call to the global state constructor will be `GlobalState(tf, 17,
 * 42-17)`.
 *
 * The stencil updater will then submit execution kernels for one or multiple passes. For each of
 * these passes, it will construct a \ref stencil::tdv::single_pass::KernelArgument "KernelArgument"
 * on the host using a reference to this global state, a reference to the SYCL handler, as well as
 * the iteration offset and number of iterations of this pass.
 */
template <typename T, typename TransFunc>
concept GlobalState =
    stencil::concepts::TransitionFunction<TransFunc> &&
    std::constructible_from<T, TransFunc, uindex_t, uindex_t> &&
    KernelArgument<typename T::KernelArgument, TransFunc> &&
    std::constructible_from<typename T::KernelArgument, T &, sycl::handler &, uindex_t, uindex_t>;

/**
 * \brief Requirements for a TDV implementation strategy.
 *
 * Such a strategy must contain a template for a valid \ref stencil::tdv::single_pass::GlobalState
 * "GlobalState", which is instantiated with the transition function and the maximal number of
 * iterations that are computed in one pass.
 */
template <typename T, typename TransFunc, uindex_t max_n_iterations>
concept Strategy =
    stencil::concepts::TransitionFunction<TransFunc> &&
    GlobalState<typename T::template GlobalState<TransFunc, max_n_iterations>, TransFunc>;

/**
 * \brief A TDV implementation strategy that inlines the TDV function into the transition function.
 *
 * This is the simplest implementation of the TDV system: The TDV construction function is called
 * every time the transition function is called; There is no precomputation done..
 *
 * For FPGA-based backends, this means that the construction function is implemented within every
 * processing element. This might be advantageous if the time-dependent value is very large and it's
 * construction is very simple. However, one could then manually merge them into the transition
 * function.
 */
struct InlineStrategy {
    template <stencil::concepts::TransitionFunction TransFunc, uindex_t max_n_iterations>
    struct GlobalState {
        using TDV = typename TransFunc::TimeDependentValue;

        GlobalState(TransFunc trans_func, uindex_t iteration_offset, uindex_t n_iterations)
            : trans_func(trans_func) {}

        struct KernelArgument {
            KernelArgument(GlobalState &global_state, sycl::handler &cgh, uindex_t iteration_offset,
                           uindex_t n_iterations)
                : trans_func(global_state.trans_func), iteration_offset(iteration_offset) {}

            using LocalState = KernelArgument;

            TDV get_time_dependent_value(uindex_t i_iteration) const {
                return trans_func.get_time_dependent_value(iteration_offset + i_iteration);
            }

          private:
            TransFunc trans_func;
            uindex_t iteration_offset;
        };

      private:
        TransFunc trans_func;
    };
};

/**
 * \brief A TDV implementation strategy that precomputes TDVs on the device.
 *
 * This precomputation is done for each pass and covers the iterations done in this pass only.
 *
 * For FPGA-based backends, this will lead to an additional for-loop prior to the main loop of the
 * execution kernel. Depending on how big the TDV is and in which way it is used by the transition
 * function, the local state may be implemented in registers or with on-chip memory.
 */
struct PrecomputeOnDeviceStrategy {
    template <stencil::concepts::TransitionFunction TransFunc, uindex_t max_n_iterations>
    struct GlobalState {
        using TDV = typename TransFunc::TimeDependentValue;

        GlobalState(TransFunc trans_func, uindex_t iteration_offset, uindex_t n_iterations)
            : trans_func(trans_func) {}

        struct KernelArgument {
            KernelArgument(GlobalState &global_state, sycl::handler &cgh, uindex_t iteration_offset,
                           uindex_t n_iterations)
                : trans_func(global_state.trans_func), iteration_offset(iteration_offset) {}

            struct LocalState {
                LocalState(KernelArgument const &kernel_argument) : values() {
                    for (uindex_t i = 0; i < max_n_iterations; i++) {
                        values[i] = kernel_argument.trans_func.get_time_dependent_value(
                            kernel_argument.iteration_offset + i);
                    }
                }

                TDV get_time_dependent_value(uindex_t i_iteration) const {
                    return values[i_iteration];
                }

              private:
                TDV values[max_n_iterations];
            };

          private:
            TransFunc trans_func;
            uindex_t iteration_offset;
        };

      private:
        TransFunc trans_func;
    };
};

/**
 * \brief A TDV implementation strategy that precomputes TDVs on the host.
 *
 * This strategy will compute all time-dependent values on the host and store them in a global
 * memory buffer. Prior to execution, the execution kernel will then load the required values into a
 * local array using a dedicated for-loop. Depending on how big the TDV is and in which way it is
 * used by the transition function, the local state may be implemented in registers or with on-chip
 * memory.
 */
struct PrecomputeOnHostStrategy {
    template <stencil::concepts::TransitionFunction TransFunc, uindex_t max_n_iterations>
    class GlobalState {
      public:
        using TDV = typename TransFunc::TimeDependentValue;

        GlobalState(TransFunc function, uindex_t iteration_offset, uindex_t n_iterations)
            : function(function), iteration_offset(iteration_offset),
              value_buffer(sycl::range<1>(n_iterations)) {
            sycl::host_accessor ac(value_buffer, sycl::read_write);
            for (uindex_t i = 0; i < n_iterations; i++) {
                ac[i] = function.get_time_dependent_value(iteration_offset + i);
            }
        }

        GlobalState(GlobalState const &other)
            : function(other.function), iteration_offset(other.iteration_offset),
              value_buffer(other.value_buffer) {}

        struct KernelArgument {
            KernelArgument(GlobalState &global_state, sycl::handler &cgh, uindex_t i_iteration,
                           uindex_t n_iterations)
                : ac() {
                assert(n_iterations <= max_n_iterations);
                assert(i_iteration >= global_state.iteration_offset);
                assert(i_iteration + n_iterations <=
                       global_state.iteration_offset + global_state.value_buffer.get_range()[0]);

                sycl::range<1> access_range(n_iterations);
                sycl::id<1> access_offset(i_iteration - global_state.iteration_offset);
                ac = sycl::accessor<TDV, 1, sycl::access::mode::read>(
                    global_state.value_buffer, cgh, access_range, access_offset);
            }

            struct LocalState {
                LocalState(KernelArgument const &kernel_argument) : values() {
                    uindex_t n_values =
                        std::min(max_n_iterations, uindex_t(kernel_argument.ac.get_range()[0]));

                    for (uindex_t i = 0; i < n_values; i++)
                        values[i] = kernel_argument.ac[i];
                }

                TDV get_time_dependent_value(uindex_t i) const { return values[i]; }

              private:
                TDV values[max_n_iterations];
            };

          private:
            sycl::accessor<TDV, 1, sycl::access::mode::read> ac;
        };

      private:
        TransFunc function;
        uindex_t iteration_offset;
        sycl::buffer<TDV, 1> value_buffer;
    };
};

} // namespace single_pass
} // namespace tdv
} // namespace stencil