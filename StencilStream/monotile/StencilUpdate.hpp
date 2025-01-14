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
#include "../Helpers.hpp"
#include "../tdv/SinglePassStrategies.hpp"
#include "Grid.hpp"
#include <chrono>
#include <type_traits>

namespace stencil {
namespace monotile {

/**
 * \brief The execution kernel of the monotile architecture
 *
 * It receives the contents of a tile and it's halo from the `in_pipe`, applies the transition
 * function when applicable and writes the result to the `out_pipe`.
 *
 * With the monotile approach, the whole grid fits in one tile. This eliminates the need to
 * calculate the cells of the tile halo, reducing the cache size and number of loop iterations. More
 * is described in \ref monotile.
 *
 * \tparam TransFunc The type of transition function to use.
 *
 * \tparam TDVKernelArgument The type of parameter for the TDV system that is passed from the host
 * to the kernel.
 *
 * \tparam n_processing_elements The number of processing elements to use. Similar to an unroll
 * factor for a loop.
 *
 * \tparam max_grid_height The maximum number of rows in the grid. This will define the size of the
 * column buffer.
 *
 * \tparam max_grid_width The maximum number of columns in the grid. This will define the bit width
 * of the cell indices.
 *
 * \tparam in_pipe The pipe to read from.
 *
 * \tparam out_pipe The pipe to write to.
 */
template <concepts::TransitionFunction TransFunc,
          tdv::single_pass::KernelArgument<TransFunc> TDVKernelArgument,
          std::size_t n_processing_elements, std::size_t vector_length, std::size_t max_grid_height,
          std::size_t max_grid_width, typename in_pipe, typename out_pipe>
    requires(n_processing_elements % TransFunc::n_subiterations == 0)
class StencilUpdateKernel {
  private:
    using Cell = typename TransFunc::Cell;
    using TDV = typename TransFunc::TimeDependentValue;
    using TDVLocalState = typename TDVKernelArgument::LocalState;
    using StencilImpl = Stencil<Cell, TransFunc::stencil_radius, TDV>;
    using CellVector = std::array<Cell, vector_length>;

    static constexpr std::size_t iters_per_pass =
        n_processing_elements / TransFunc::n_subiterations;

    static constexpr std::size_t max_vect_grid_width = int_ceil_div(max_grid_width, vector_length);

    // Round up the stencil buffer lead to the next integer multiple of the vector length.
    static constexpr std::size_t vect_stencil_buffer_lead =
        int_ceil_div(TransFunc::stencil_radius, vector_length);
    static constexpr std::size_t stencil_buffer_lead = vect_stencil_buffer_lead * vector_length;
    static constexpr std::size_t stencil_buffer_height = 2 * TransFunc::stencil_radius + 1;
    static constexpr std::size_t stencil_buffer_width =
        TransFunc::stencil_radius + vector_length + stencil_buffer_lead;

    template <typename T> static constexpr T calc_pipeline_latency(T grid_width) {
        T vect_grid_width = int_ceil_div<T>(grid_width, vector_length);
        return T(n_processing_elements * TransFunc::stencil_radius) * vect_grid_width +
               T(n_processing_elements * vect_stencil_buffer_lead);
    }

    template <typename T> static constexpr T calc_n_steps(T grid_height, T grid_width) {
        T vect_grid_width = int_ceil_div<T>(grid_width, vector_length);
        return grid_height * vect_grid_width + calc_pipeline_latency<T>(grid_width);
    }

    using index_r_t = ac_int<std::bit_width(max_grid_height) + 1, true>;
    using uindex_r_t = ac_int<std::bit_width(max_grid_height), false>;

    using uindex_c_t = ac_int<std::bit_width(max_grid_width), false>;

    using index_vect_c_t = ac_int<std::bit_width(max_vect_grid_width) + 1, true>;
    using uindex_vect_c_t = ac_int<std::bit_width(max_vect_grid_width), false>;

    using uindex_step_t =
        ac_int<std::bit_width(calc_n_steps(max_grid_height, max_grid_width)), false>;

  public:
    /**
     * \brief Create and configure the execution kernel.
     *
     * \param trans_func The instance of the transition function to use.
     *
     * \param i_iteration The iteration index of the input cells.
     *
     * \param target_i_iteration The final, requested iteration index after the updates. This may be
     * higher than what the kernel can process in one pass. In this case, the kernel will compute
     * the maximum number of iterations.
     *
     * \param grid_height The number of cell rows in the grid.
     *
     * \param grid_width The number of cell columns in the grid.
     *
     * \param halo_value The value of cells outside the grid.
     *
     * \param tdv_kernel_argument The argument for the TDV system that is passed from the host to
     * the device. This may for example contain global memory accessors.
     */
    StencilUpdateKernel(TransFunc trans_func, std::size_t i_iteration,
                        std::size_t target_i_iteration, std::size_t grid_height,
                        std::size_t grid_width, Cell halo_value,
                        TDVKernelArgument tdv_kernel_argument)
        : trans_func(trans_func), i_iteration(i_iteration), target_i_iteration(target_i_iteration),
          grid_height(grid_height), grid_width(grid_width), halo_value(halo_value),
          tdv_kernel_argument(tdv_kernel_argument) {
        assert(grid_width >= 2);
        assert(TransFunc::stencil_radius <= grid_width && grid_width <= max_grid_width);
        assert(TransFunc::stencil_radius <= grid_height && grid_height <= max_grid_height);
        assert(i_iteration < target_i_iteration);
    }

    /**
     * \brief Execute the kernel.
     */
    void operator()() const {
        uindex_vect_c_t vect_grid_width = int_ceil_div<uindex_c_t>(grid_width, vector_length);

        [[intel::fpga_register]] index_r_t r[n_processing_elements];
        [[intel::fpga_register]] uindex_vect_c_t vect_c[n_processing_elements];
        TDVLocalState tdv_local_state(tdv_kernel_argument);

        // Initializing (output) column and row counters.
        index_r_t prev_r = 0;
        index_vect_c_t prev_vect_c = 0;
#pragma unroll
        for (std::size_t i = 0; i < n_processing_elements; i++) {
            prev_r -= TransFunc::stencil_radius;
            prev_vect_c -= vect_stencil_buffer_lead;
            if (prev_vect_c < 0) {
                prev_vect_c += vect_grid_width;
                prev_r -= 1;
            }
            r[i] = prev_r;
            vect_c[i] = prev_vect_c;
        }

        /*
         * The intel::numbanks attribute requires a power of two as it's argument and if the
         * number of processing elements isn't a power of two, it would produce an error. Therefore,
         * we calculate the next power of two and use it to allocate the cache. The compiler is
         * smart enough to see that these additional banks in the cache aren't used and therefore
         * optimizes them away.
         */
        [[intel::fpga_memory,
          intel::numbanks(2 * std::bit_ceil(n_processing_elements))]] Padded<CellVector>
            cache[2][max_vect_grid_width][std::bit_ceil(n_processing_elements)]
                 [stencil_buffer_height - 1];
        [[intel::fpga_register]] Cell stencil_buffer[n_processing_elements][stencil_buffer_height]
                                                    [stencil_buffer_width];


        /*
         * OneAPI 2024.1 and newer finds a WAR memory dependency on the cache that it can't resolve
         * on its own. This issue is resolved by declaring that the distance between a read and a
         * write to the same memory location is at least two. This is ensured by requiring a minimal
         * grid width of two.
         */
        [[intel::ivdep(2, cache)]]
        for (uindex_step_t i = 0; i < calc_n_steps<uindex_step_t>(grid_height, grid_width); i++) {
            CellVector carry;
            if (i < uindex_r_t(grid_height) * vect_grid_width) {
                carry = in_pipe::read();
            } else {
#pragma unroll
                for (std::size_t i_cell = 0; i_cell < vector_length; i_cell++) {
                    carry[i_cell] = halo_value;
                }
            }

#pragma unroll
            for (std::size_t i_processing_element = 0; i_processing_element < n_processing_elements;
                 i_processing_element++) {
#pragma unroll
                for (std::size_t r = 0; r < stencil_buffer_height; r++) {
#pragma unroll
                    for (std::size_t c = 0; c < stencil_buffer_width - vector_length; c++) {
                        stencil_buffer[i_processing_element][r][c] =
                            stencil_buffer[i_processing_element][r][c + vector_length];
                    }
                }

                // Update the stencil buffer and cache with previous cache contents and the new
                // input cell.
#pragma unroll
                for (std::size_t cache_r = 0; cache_r < stencil_buffer_height; cache_r++) {
                    CellVector new_vect;
                    if (cache_r == stencil_buffer_height - 1) {
                        new_vect = carry;
                    } else {
                        new_vect = cache[r[i_processing_element][0]][vect_c[i_processing_element]]
                                        [i_processing_element][cache_r]
                                            .value;
                    }

#pragma unroll
                    for (std::size_t i_vector_cell = 0; i_vector_cell < vector_length;
                         i_vector_cell++) {
                        stencil_buffer[i_processing_element][cache_r]
                                      [TransFunc::stencil_radius + stencil_buffer_lead +
                                       i_vector_cell] = new_vect[i_vector_cell];
                    }

                    if (cache_r > 0) {
                        cache[(~r[i_processing_element])[0]][vect_c[i_processing_element]]
                             [i_processing_element][cache_r - 1]
                                 .value = new_vect;
                    }
                }

                std::size_t pe_iteration = i_processing_element / TransFunc::n_subiterations;
                std::size_t n_iterations =
                    std::min(target_i_iteration - i_iteration, iters_per_pass);

                if (pe_iteration < n_iterations) {
                    TDV tdv = tdv_local_state.get_time_dependent_value(pe_iteration);

                    bool v_halo_mask[stencil_buffer_height];
#pragma unroll
                    for (std::size_t mask_i = 0; mask_i < stencil_buffer_height; mask_i++) {
                        std::ptrdiff_t cell_r = std::ptrdiff_t(r[i_processing_element]) +
                                                std::ptrdiff_t(mask_i - TransFunc::stencil_radius);
                        v_halo_mask[mask_i] = cell_r >= 0 && cell_r < grid_height;
                    }

                    bool h_halo_mask[2 * TransFunc::stencil_radius + vector_length];
#pragma unroll
                    for (std::size_t mask_i = 0;
                         mask_i < 2 * TransFunc::stencil_radius + vector_length; mask_i++) {
                        std::ptrdiff_t c =
                            std::ptrdiff_t(vect_c[i_processing_element]) * vector_length -
                            TransFunc::stencil_radius + mask_i;
                        h_halo_mask[mask_i] = c >= 0 && c < grid_width;
                    }

#pragma unroll
                    for (std::size_t i_vector_cell = 0; i_vector_cell < vector_length;
                         i_vector_cell++) {
                        std::size_t c = std::size_t(vect_c[i_processing_element]) * vector_length +
                                        i_vector_cell;

                        StencilImpl stencil(sycl::id<2>(r[i_processing_element], c),
                                            sycl::range<2>(grid_height, grid_width),
                                            i_iteration + pe_iteration,
                                            i_processing_element % TransFunc::n_subiterations, tdv);

#pragma unroll
                        for (std::size_t cell_r = 0; cell_r < 2 * TransFunc::stencil_radius + 1;
                             cell_r++) {
#pragma unroll
                            for (std::size_t cell_c = 0; cell_c < 2 * TransFunc::stencil_radius + 1;
                                 cell_c++) {
                                if (v_halo_mask[cell_r] && h_halo_mask[cell_c + i_vector_cell]) {
                                    stencil[sycl::id<2>(cell_r, cell_c)] =
                                        stencil_buffer[i_processing_element][cell_r]
                                                      [cell_c + i_vector_cell];
                                } else {
                                    stencil[sycl::id<2>(cell_r, cell_c)] = halo_value;
                                }
                            }
                        }

                        carry[i_vector_cell] = trans_func(stencil);
                    }
                } else {
#pragma unroll
                    for (std::size_t i_vector_cell = 0; i_vector_cell < vector_length;
                         i_vector_cell++) {
                        carry[i_vector_cell] =
                            stencil_buffer[i_processing_element][TransFunc::stencil_radius]
                                          [TransFunc::stencil_radius + i_vector_cell];
                    }
                }

                vect_c[i_processing_element] += 1;
                if (vect_c[i_processing_element] == vect_grid_width) {
                    vect_c[i_processing_element] = 0;
                    r[i_processing_element] += 1;
                }
            }

            if (i >= calc_pipeline_latency<uindex_step_t>(grid_width)) {
                out_pipe::write(carry);
            }
        }
    }

  private:
    TransFunc trans_func;
    std::size_t i_iteration;
    std::size_t target_i_iteration;
    std::size_t grid_height;
    std::size_t grid_width;
    Cell halo_value;
    TDVKernelArgument tdv_kernel_argument;
};

/**
 * \brief A grid updater that applies an iterative stencil code to a grid.
 *
 * This updater applies an iterative stencil code, defined by the template parameter `F`, to the
 * grid; As often as requested. Since the underlying FPGA design follows the Monotile architecture
 * (See \ref monotile), an instance of this updater template can only process grids up to the
 * defined `max_grid_width` and `max_grid_height`.
 *
 * \tparam F The transition function to apply to input grids.
 *
 * \tparam n_processing_elements (Optimization parameter) The number of processing elements (PEs) to
 * implement. Increasing the number of PEs leads to a higher performance since more iterations are
 * computed in parallel. However, it will also increase the resource and space usage of the design.
 * Too many PEs might also decrease the clock frequency.
 *
 * \tparam max_grid_height (Optimization parameter) The maximally supported grid height. For best
 * hardware utilization, this should be a power of two. Increase this parameter to the maximum your
 * application is expected to handle. However, higher maximal grid height might lead to increased
 * logic and space usage as well as decreased clock frequencies.
 *
 * \tparam max_grid_width (Optimization parameter) The maximally supported grid width. Increase
 * this parameter to the maximum your application is expected to handle. However, increasing the
 * maximal grid width will increase the BRAM usage of each PE.
 *
 * \tparam TDVStrategy (Optimization parameter) The precomputation strategy for the time-dependent
 * value system.
 *
 * \tparam word_size (Optimization parameter) The width of the global memory channel, in bytes. For
 * DDR-based systems, this should be 512 bits, or 64 bytes.
 */
template <concepts::TransitionFunction F, std::size_t n_processing_elements = 1,
          std::size_t vector_length = 1, std::size_t max_grid_height = 1024,
          std::size_t max_grid_width = 1024,
          tdv::single_pass::Strategy<F, n_processing_elements> TDVStrategy =
              tdv::single_pass::InlineStrategy>
class StencilUpdate {
  private:
    using Cell = F::Cell;
    using TDV = typename F::TimeDependentValue;

  public:
    /// \brief Shorthand for the used and supported grid type.
    using GridImpl = Grid<Cell, vector_length>;

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
         *
         * For some setups, it might be necessary to explicitly select the device to use for
         * computation. This can be done for example with the `sycl::ext::intel::fpga_selector_v`
         * class in the `sycl/ext/intel/fpga_extensions.hpp` header. This selector will select the
         * first FPGA it sees.
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
         * \brief Enable profiling.
         *
         * Setting this option to true will enable the recording of computation start and end
         * timestamps. The recorded kernel runtime can be fetched using the \ref
         * StencilUpdate::get_kernel_runtime method.
         */
        bool profiling = false;
    };

    /**
     * \brief Create a new stencil updater object.
     */
    StencilUpdate(Params params)
        : params(params), n_processed_cells(0), work_events(), walltime(0.0) {}

    /**
     * \brief Return a reference to the parameters.
     *
     * Modifications to the parameters struct will be used in the next call to \ref operator()().
     */
    Params &get_params() { return params; }

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
        if (source_grid.get_grid_width() < 2) {
            throw std::range_error("The grid is too narrow. The monotile backend can only process "
                                   "grids with at least two columns.");
        }
        if (source_grid.get_grid_height() > max_grid_height) {
            throw std::range_error("The grid is too tall for the stencil update kernel.");
        }
        if (source_grid.get_grid_width() > max_grid_width) {
            throw std::range_error("The grid is too wide for the stencil update kernel.");
        }
        using in_pipe = sycl::pipe<class monotile_in_pipe, std::array<Cell, vector_length>>;
        using out_pipe = sycl::pipe<class monotile_out_pipe, std::array<Cell, vector_length>>;

        constexpr std::size_t iters_per_pass = n_processing_elements / F::n_subiterations;

        using TDVGlobalState = TDVStrategy::template GlobalState<F, iters_per_pass>;
        using TDVKernelArgument = typename TDVGlobalState::KernelArgument;
        using ExecutionKernelImpl =
            StencilUpdateKernel<F, TDVKernelArgument, n_processing_elements, vector_length,
                                max_grid_height, max_grid_width, in_pipe, out_pipe>;

        sycl::queue input_kernel_queue =
            sycl::queue(params.device, {sycl::property::queue::in_order{}});
        sycl::queue output_kernel_queue =
            sycl::queue(params.device, {sycl::property::queue::in_order{}});

        sycl::queue update_kernel_queue =
            sycl::queue(params.device, {cl::sycl::property::queue::enable_profiling{},
                                        sycl::property::queue::in_order{}});

        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();

        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        F trans_func = params.transition_function;
        TDVGlobalState tdv_global_state(trans_func, params.iteration_offset, params.n_iterations);

        auto walltime_start = std::chrono::high_resolution_clock::now();

        std::size_t target_n_iterations = params.iteration_offset + params.n_iterations;
        for (std::size_t i = params.iteration_offset; i < target_n_iterations;
             i += iters_per_pass) {
            pass_source->template submit_read<in_pipe, max_grid_height, max_grid_width>(
                input_kernel_queue);
            std::size_t iters_in_this_pass = std::min(iters_per_pass, target_n_iterations - i);

            sycl::event work_event = update_kernel_queue.submit([&](sycl::handler &cgh) {
                TDVKernelArgument tdv_kernel_argument(tdv_global_state, cgh, i, iters_in_this_pass);
                ExecutionKernelImpl exec_kernel(
                    trans_func, i, target_n_iterations, source_grid.get_grid_height(),
                    source_grid.get_grid_width(), params.halo_value, tdv_kernel_argument);
                cgh.single_task<ExecutionKernelImpl>(exec_kernel);
            });
            if (params.profiling) {
                work_events.push_back(work_event);
            }

            pass_target->template submit_write<out_pipe, max_grid_height, max_grid_width>(
                output_kernel_queue);

            if (i == params.iteration_offset) {
                pass_source = &swap_grid_b;
                pass_target = &swap_grid_a;
            } else {
                std::swap(pass_source, pass_target);
            }
        }

        if (params.blocking) {
            output_kernel_queue.wait();
        }

        auto walltime_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> walltime = walltime_end - walltime_start;
        this->walltime += walltime.count();

        n_processed_cells +=
            params.n_iterations * source_grid.get_grid_height() * source_grid.get_grid_width();

        return *pass_source;
    }

    /**
     * \brief Return the accumulated total number of cells processed by this updater.
     *
     * For each call of to \ref operator()(), this is the width times the height of the grid, times
     * the number of computed iterations. This will also be accumulated across multiple calls to
     * \ref operator()().
     */
    std::size_t get_n_processed_cells() const { return n_processed_cells; }

    /**
     * \brief Return the accumulated total runtime of the execution kernel.
     *
     * This runtime is accumulated across multiple calls to \ref operator()(). However, this is only
     * possible if \ref Params::profiling is set to true.
     */
    double get_kernel_runtime() const {
        double kernel_runtime = 0.0;
        for (sycl::event work_event : work_events) {
            const double timesteps_per_second = 1000000000.0;
            double start =
                double(work_event
                           .get_profiling_info<cl::sycl::info::event_profiling::command_start>()) /
                timesteps_per_second;
            double end =
                double(
                    work_event.get_profiling_info<cl::sycl::info::event_profiling::command_end>()) /
                timesteps_per_second;
            kernel_runtime += end - start;
        }
        return kernel_runtime;
    }

    /**
     * \brief Return the accumulated runtime of the updater, measured from the host side.
     *
     * For each call to \ref operator()(), the time it took to submit all kernels and, if \ref
     * Params::blocking is true, to finish the computation is recorded and accumulated.
     */
    double get_walltime() const { return walltime; }

  private:
    Params params;
    std::size_t n_processed_cells;
    double walltime;
    std::vector<sycl::event> work_events;
};

} // namespace monotile
} // namespace stencil