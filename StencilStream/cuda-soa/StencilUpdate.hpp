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
#include "../Concepts.hpp"
#include "../Stencil.hpp"
#include "Grid.hpp"
#include <chrono>
#include <cstddef>
#include <utility>
#include <vector>

template <typename T> struct cell_members;

namespace stencil {
namespace cuda {

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

/**
 * \brief A grid updater that applies an iterative stencil code to a grid.
 *
 * This updater applies an iterative stencil code, defined by the template parameter `F`, to the
 * grid; As often as requested.
 *
 * \tparam F The transition function to apply to input grids.
 */
template <concepts::TransitionFunction F> class StencilUpdate {
  private:
    using Cell = F::Cell;
    std::vector<sycl::event> work_events;

  public:
    /// \brief Shorthand for the used and supported grid type.
    using GridImpl = Grid<Cell>;

    /**
     * \brief Parameters for the stencil updater.
     */
    struct Params {
        /**
         * \brief An instance of the transition function type.
         *
         * User applications may store runtime parameters here.
         */
        F transition_function;

        /**
         *  \brief The cell value to present for cells outside of the grid.
         */
        Cell halo_value = Cell();

        /**
         * \brief The iteration index offset.
         *
         * This offset will be added to the "actual" iteration index. This way, simulations can
         * "resume" with the next timestep if the intermediate grid has been evaluated by the host.
         */
        std::size_t iteration_offset = 0;

        /**
         * \brief The number of iterations to compute.
         */
        std::size_t n_iterations = 1;

        /**
         * \brief The device to use for computations.
         */
        sycl::device device = sycl::device();

        /**
         * \brief Should the stencil updater block until completion, or return immediately after all
         * kernels have been submitted.
         *
         * Choosing one option or the other won't effect the correctness: For example, if you choose
         * a non-blocking stencil updater and immediately try to access the grid after the updater
         * has returned, SYCL/OneAPI will block your thread until the computations are complete and
         * it can actually provide you access to the data.
         */
        bool blocking = false;

        /**
         * \brief The kernel time kan be profiled
         */
        bool profiling = true;
    };

    /**
     * \brief Create a new stencil updater object.
     */
    StencilUpdate(Params params)
        : params(params), n_processed_cells(0), walltime(0.0), work_events() {}

    /**
     * \brief Compute a new grid based on the source grid, using the configured transition function.
     *
     * The computation does not work in-place. Instead, it will allocate two additional grids with
     * the same size as the source grid and use them for a double buffering scheme. Therefore, you
     * are free to reuse the source grid as it will not be altered.
     *
     * If \ref Params::blocking is set to true, this method will block until the computation is
     * complete. Otherwise, it will return as soon as all kernels are submitted.
     */

    GridImpl operator()(GridImpl &source_grid) {
        // Create a SYCL queue for the scatter/gather operations with profiling enabled
        sycl::queue queue_scatter_gather =
            sycl::queue(params.device, {sycl::property::queue::enable_profiling()});

        // Submit a kernel to scatter the data from the source Grid into individual buffers
        queue_scatter_gather.submit([&](sycl::handler &cgh) {
            sycl::accessor ac_source_grid(source_grid.get_buffer(), cgh);

            // Create a tuple of write-only accessors for each member buffer in the grid
            auto acc_tuple = std::apply(
                [&](auto &...buf) {
                    return std::make_tuple(sycl::accessor(buf, cgh, sycl::write_only)...);
                },
                source_grid.get_buffers());

            cgh.parallel_for(ac_source_grid.get_range(), [=](sycl::id<2> id) {
                Cell cell = ac_source_grid[id[0]][id[1]];
                size_t cell_id = id[0] * ac_source_grid.get_range()[1] + id[1];

                // For each member of Cell, write its value to the corresponding buffer
                for_each_in_two_tuples(
                    acc_tuple, cell_members<Cell>::fields, [&](auto &acc, auto member_ptr) {
                        acc[cell_id] = static_cast<std::remove_reference_t<decltype(acc[cell_id])>>(
                            cell.*member_ptr);
                    });
            });
        });

        queue_scatter_gather.wait();
        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();
        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        sycl::queue queue(params.device);
        auto walltime_start = std::chrono::high_resolution_clock::now();

        for (std::size_t i_iter = 0; i_iter < params.n_iterations; i_iter++) {
            for (std::size_t i_subiter = 0; i_subiter < F::n_subiterations; i_subiter++) {
                run_iter(queue, pass_source, pass_target, params.iteration_offset + i_iter,
                         i_subiter);
                if (i_iter == 0 && i_subiter == 0) {
                    pass_source = &swap_grid_b;
                    pass_target = &swap_grid_a;
                } else {
                    std::swap(pass_source, pass_target);
                }
            }
        }

        if (params.blocking) {
            queue.wait();
        }

        auto walltime_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> walltime = walltime_end - walltime_start;
        this->walltime += walltime.count();

        // Submit a kernel to gather data from the separate member buffers back into the Grid
        queue_scatter_gather.submit([&](sycl::handler &cgh) {
            sycl::accessor ac_pass_grid(pass_source->get_buffer(), cgh, sycl::write_only);

            // Create a tuple of read-only accessors for each member buffer
            auto acc_tuple = std::apply(
                [&](auto &...buf) {
                    return std::make_tuple(sycl::accessor(buf, cgh, sycl::read_only)...);
                },
                pass_source->get_buffers());

            cgh.parallel_for(ac_pass_grid.get_range(), [=](sycl::id<2> id) {
                Cell cell;
                size_t cell_id = id[0] * ac_pass_grid.get_range()[1] + id[1];

                // For each member of Cell, read its value from the corresponding buffer
                for_each_in_two_tuples(
                    acc_tuple, cell_members<Cell>::fields, [&](auto &acc, auto member_ptr) {
                        cell.*member_ptr =
                            static_cast<std::remove_reference_t<decltype(acc[cell_id])>>(
                                acc[cell_id]);
                    });

                ac_pass_grid[id[0]][id[1]] = cell;
            });
        });
        queue_scatter_gather.wait();

        n_processed_cells +=
            params.n_iterations * source_grid.get_grid_height() * source_grid.get_grid_width();

        return *pass_source;
    }

    /**
     * \brief Return a reference to the parameters.
     *
     * Modifications to the parameters struct will be used in the next call to \ref operator()().
     */
    Params &get_params() { return params; }

    /**
     * \brief Return the accumulated total number of cells processed by this updater.
     *
     * For each call of to \ref operator()(), this is the width times the height of the grid, times
     * the number of computed iterations. This will also be accumulated across multiple calls to
     * \ref operator()().
     */
    std::size_t get_n_processed_cells() const { return n_processed_cells; }

    /**
     * \brief Return the accumulated runtime of the updater, measured from the host side.
     *
     * For each call to \ref operator()(), the time it took to submit all kernels and, if \ref
     * Params::blocking is true, to finish the computation is recorded and accumulated.
     */
    double get_walltime() const { return walltime; }

    /**
     * \brief Return the total runtime of all kernels submitted by the updater.
     *
     * This function sums the execution time of each kernel recorded in the `work_events` vector.
     * For every `sycl::event` in `work_events`, the start and end timestamps are queried using the
     * SYCL profiling API (`command_start` and `command_end`). The difference between end and start
     * gives the kernel execution time in seconds. These times are accumulated to produce the
     * total kernel runtime.
     *
     * \return The total time spent executing kernels on the device, in seconds.
     *
     * \note This only measures the device-side kernel execution. It does not include time spent
     *       on host-side work, queue submission, or any synchronization outside the event.
     */
    double get_kernel_runtime() const {
        double kernel_runtime = 0.0;
        for (sycl::event work_event : work_events) {
            const double timesteps_per_second = 1000000000.0;
            double start =
                double(
                    work_event.get_profiling_info<sycl::info::event_profiling::command_start>()) /
                timesteps_per_second;
            double end =
                double(work_event.get_profiling_info<sycl::info::event_profiling::command_end>()) /
                timesteps_per_second;
            kernel_runtime += end - start;
        }
        return kernel_runtime;
    }

  private:
    /**
     * \brief Update the source grid by one iteration.
     *
     * This method will read the current state of the grid from the pass source and write the update
     * the pass target.
     *
     * \param queue The queue to submit the kernel to.
     *
     * \param pass_source A pointer to a grid. The old state of the grid will be read from here.
     *
     * \param pass_target A pointer to a grid. The new state will be written to this grid.
     *
     * \param i_iter The index of the iteration to compute.
     *
     * \param i_subiter The index of the sub-iteration to compute.
     */
    void run_iter(sycl::queue queue, GridImpl *pass_source, GridImpl *pass_target,
                  std::size_t i_iter, std::size_t i_subiter) {
        using TDV = typename F::TimeDependentValue;
        using StencilImpl = Stencil<Cell, F::stencil_radius, TDV>;

        sycl::event work_event = queue.submit([&](sycl::handler &cgh) {
            sycl::accessor source_ac(pass_source->get_buffer(), cgh, sycl::read_only);
            sycl::accessor target_ac(pass_target->get_buffer(), cgh, sycl::write_only);

            auto acc_tuple_pass_source = std::apply(
                [&](auto &...buf) {
                    return std::make_tuple(sycl::accessor(buf, cgh, sycl::read_only)...);
                },
                pass_source->get_buffers());

            auto acc_tuple_pass_target = std::apply(
                [&](auto &...buf) {
                    return std::make_tuple(sycl::accessor(buf, cgh, sycl::write_only)...);
                },
                pass_target->get_buffers());

            std::size_t grid_height = source_ac.get_range()[0];
            std::size_t grid_width = source_ac.get_range()[1];
            Cell halo_value = params.halo_value;
            F transition_function = params.transition_function;
            TDV tdv = transition_function.get_time_dependent_value(i_iter);

            auto kernel = [=](sycl::id<2> id) {
                StencilImpl stencil(id, source_ac.get_range(), i_iter, i_subiter, tdv);

                for (std::size_t rel_r = 0; rel_r < 2 * F::stencil_radius + 1; rel_r++) {
                    for (std::size_t rel_c = 0; rel_c < 2 * F::stencil_radius + 1; rel_c++) {
                        Cell cell;
                        if (id[0] + rel_r >= F::stencil_radius &&
                            id[1] + rel_c >= F::stencil_radius &&
                            id[0] + rel_r < grid_height + F::stencil_radius &&
                            id[1] + rel_c < grid_width + F::stencil_radius) {
                            size_t cell_id = (id[0] + rel_r - F::stencil_radius) * grid_width +
                                             (id[1] + rel_c - F::stencil_radius);
                            for_each_in_two_tuples(
                                acc_tuple_pass_source, cell_members<Cell>::fields,
                                [&](auto &acc, auto member_ptr) {
                                    cell.*member_ptr = static_cast<
                                        std::remove_reference_t<decltype(acc[cell_id])>>(
                                        acc[cell_id]);
                                });

                        } else {
                            cell = halo_value;
                        }
                        stencil[sycl::id<2>(rel_r, rel_c)] = cell;
                    }
                }

                size_t cell_id = id[0] * grid_width + id[1];
                Cell new_cell = transition_function(stencil);

                for_each_in_two_tuples(
                    acc_tuple_pass_target, cell_members<Cell>::fields,
                    [&](auto &acc, auto member_ptr) {
                        acc[cell_id] = static_cast<std::remove_reference_t<decltype(acc[cell_id])>>(
                            new_cell.*member_ptr);
                    });
            };

            cgh.parallel_for(source_ac.get_range(), kernel);
        });
        if (params.profiling) {
            work_events.push_back(work_event);
        }
    }

    Params params;
    std::size_t n_processed_cells;
    double walltime;
};
} // namespace cuda
} // namespace stencil