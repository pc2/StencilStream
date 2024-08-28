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
#include "../GenericID.hpp"
#include "../Helpers.hpp"
#include "../Index.hpp"
#include "../tdv/SinglePassStrategies.hpp"
#include "Grid.hpp"

#include <chrono>
#include <optional>

namespace stencil {
namespace tiling {

/**
 * \brief A kernel that executes a stencil transition function on a tile.
 *
 * It receives the contents of a tile and it's halo from the `in_pipe`, applies the transition
 * function when applicable and writes the result to the `out_pipe`.
 *
 * \tparam TransFunc The type of transition function to use.
 *
 * \tparam TDVKernelArgument The type of parameter for the TDV system that is passed from the host
 * to the kernel.
 *
 * \tparam n_processing_elements The number of processing elements to use. Similar to an unroll
 * factor for a loop.
 *
 * \tparam output_tile_width The number of columns in a grid tile.
 *
 * \tparam output_tile_height The number of rows in a grid tile.
 *
 * \tparam in_pipe The pipe to read from.
 *
 * \tparam out_pipe The pipe to write to.
 */
template <concepts::TransitionFunction TransFunc,
          tdv::single_pass::KernelArgument<TransFunc> TDVKernelArgument,
          uindex_t n_processing_elements, uindex_t output_tile_width, uindex_t output_tile_height,
          typename in_pipe, typename out_pipe>
    requires(n_processing_elements % TransFunc::n_subiterations == 0)
class StencilUpdateKernel {
  private:
    using Cell = typename TransFunc::Cell;
    using TDV = typename TransFunc::TimeDependentValue;
    using StencilImpl = Stencil<Cell, TransFunc::stencil_radius, TDV>;
    using TDVLocalState = typename TDVKernelArgument::LocalState;

    static constexpr uindex_t stencil_diameter = StencilImpl::diameter;

    static constexpr uindex_t halo_radius = TransFunc::stencil_radius * n_processing_elements;

    static constexpr uindex_t max_input_tile_width = 2 * halo_radius + output_tile_width;

    static constexpr uindex_t input_tile_height = 2 * halo_radius + output_tile_height;

    static constexpr uindex_t n_input_cells = max_input_tile_width * input_tile_height;

    using index_stencil_t = typename StencilImpl::index_stencil_t;
    using uindex_stencil_t = typename StencilImpl::uindex_stencil_t;
    using StencilID = typename StencilImpl::StencilID;
    using StencilUID = typename StencilImpl::StencilUID;

    static constexpr unsigned long bits_1d =
        std::bit_width(std::max(max_input_tile_width, input_tile_height));
    using index_1d_t = ac_int<bits_1d + 1, true>;
    using uindex_1d_t = ac_int<bits_1d, false>;

    static constexpr unsigned long bits_2d = 2 * bits_1d;
    using index_2d_t = ac_int<bits_2d + 1, true>;
    using uindex_2d_t = ac_int<bits_2d, false>;

    // bits_pes must not be 1 since modulo operations on ac_int<1, false> don't work with oneAPI
    // 2022.2. If they work with a future version of oneAPI, this can be reverted.
    static constexpr unsigned long bits_pes =
        std::max<int>(2, std::bit_width(n_processing_elements));
    using index_pes_t = ac_int<bits_pes + 1, true>;
    using uindex_pes_t = ac_int<bits_pes, false>;

  public:
    /**
     * \brief Create and configure the execution kernel.
     *
     * \param trans_func The instance of the transition function to use.
     *
     * \param i_iteration The iteration index of the input cells.
     *
     * \param target_i_iteration The number of iterations to compute. If this number is bigger
     * than `n_processing_elements`, only `n_processing_elements` iterations will be computed.
     *
     * \param grid_c_offset The column offset of the processed tile relative to the grid's origin,
     * not including the halo. For example, for the most north-western tile the offset will always
     * be (0,0), not (-halo_radius,-halo_radius)
     *
     * \param grid_r_offset The row offset of the processed tile relative to the grid's origin. See
     * `grid_c_offset` for details.
     *
     * \param grid_width The number of cell columns in the grid.
     *
     * \param grid_height The number of cell rows in the grid.
     *
     * \param halo_value The value of cells in the grid halo.
     *
     * \param tdv_kernel_argument The argument for the TDV system that is passed from the host to
     * the device. This may for example contain global memory accessors.
     */
    StencilUpdateKernel(TransFunc trans_func, uindex_t i_iteration, uindex_t target_i_iteration,
                        uindex_t grid_c_offset, uindex_t grid_r_offset, uindex_t grid_width,
                        uindex_t grid_height, Cell halo_value,
                        TDVKernelArgument tdv_kernel_argument)
        : trans_func(trans_func), i_iteration(i_iteration), target_i_iteration(target_i_iteration),
          grid_c_offset(grid_c_offset), grid_r_offset(grid_r_offset), grid_width(grid_width),
          grid_height(grid_height), halo_value(halo_value),
          tdv_kernel_argument(tdv_kernel_argument) {
        assert(grid_c_offset % output_tile_width == 0);
        assert(grid_r_offset % output_tile_height == 0);
    }

    /**
     * \brief Execute the configured operations.
     */
    void operator()() const {
        TDVLocalState tdv_local_state(tdv_kernel_argument);

        uindex_1d_t input_tile_c = 0;
        uindex_1d_t input_tile_r = 0;

        /*
         * The intel::numbanks attribute requires a power of two as it's argument and if the
         * number of processing elements isn't a power of two, it would produce an error. Therefore,
         * we calculate the next power of two and use it to allocate the cache. The compiler is
         * smart enough to see that these additional banks in the cache aren't used and therefore
         * optimizes them away.
         */
        [[intel::fpga_memory,
          intel::numbanks(2 * std::bit_ceil(n_processing_elements))]] Padded<Cell>
            cache[2][input_tile_height][std::bit_ceil(n_processing_elements)][stencil_diameter - 1];
        [[intel::fpga_register]] Cell stencil_buffer[n_processing_elements][stencil_diameter]
                                                    [stencil_diameter];

        uindex_1d_t output_tile_section_width =
            std::min(output_tile_width, grid_width - grid_c_offset);
        uindex_1d_t output_tile_section_height =
            std::min(output_tile_height, grid_height - grid_r_offset);
        uindex_1d_t input_tile_section_width = output_tile_section_width + 2 * halo_radius;
        uindex_1d_t input_tile_section_height = output_tile_section_height + 2 * halo_radius;
        uindex_2d_t n_iterations = input_tile_section_width * input_tile_section_height;

        for (uindex_2d_t i = 0; i < n_iterations; i++) {
            [[intel::fpga_register]] Cell carry = in_pipe::read();

#pragma unroll
            for (uindex_pes_t i_processing_element = 0;
                 i_processing_element < uindex_pes_t(n_processing_elements);
                 i_processing_element++) {
                /*
                 * Shift up every value in the stencil_buffer.
                 * This operation does not touch the values in the bottom row, which will be filled
                 * from the cache and the new input value later.
                 */
#pragma unroll
                for (uindex_stencil_t r = 0; r < uindex_stencil_t(stencil_diameter - 1); r++) {
#pragma unroll
                    for (uindex_stencil_t c = 0; c < uindex_stencil_t(stencil_diameter); c++) {
                        stencil_buffer[i_processing_element][c][r] =
                            stencil_buffer[i_processing_element][c][r + 1];
                    }
                }

                index_1d_t rel_input_grid_c =
                    index_1d_t(input_tile_c) -
                    index_1d_t((stencil_diameter - 1) +
                               (n_processing_elements + i_processing_element - 2) *
                                   TransFunc::stencil_radius);
                index_t input_grid_c = grid_c_offset + rel_input_grid_c.to_int64();
                index_1d_t rel_input_grid_r =
                    index_1d_t(input_tile_r) -
                    index_1d_t((stencil_diameter - 1) +
                               (n_processing_elements + i_processing_element - 2) *
                                   TransFunc::stencil_radius);
                index_t input_grid_r = grid_r_offset + rel_input_grid_r.to_int64();

                // Update the stencil buffer and cache with previous cache contents and the new
                // input cell.
#pragma unroll
                for (uindex_stencil_t cache_c = 0; cache_c < uindex_stencil_t(stencil_diameter);
                     cache_c++) {
                    Cell new_value;
                    if (cache_c == uindex_stencil_t(stencil_diameter - 1)) {
                        bool is_halo = (grid_c_offset == 0 && rel_input_grid_c < 0);
                        is_halo |= (grid_r_offset == 0 && rel_input_grid_r < 0);
                        is_halo |= input_grid_c >= grid_width || input_grid_r >= grid_height;

                        new_value = is_halo ? halo_value : carry;
                    } else {
                        new_value =
                            cache[input_tile_c[0]][input_tile_r][i_processing_element][cache_c]
                                .value;
                    }

                    stencil_buffer[i_processing_element][cache_c][stencil_diameter - 1] = new_value;
                    if (cache_c > 0) {
                        cache[(~input_tile_c)[0]][input_tile_r][i_processing_element][cache_c - 1]
                            .value = new_value;
                    }
                }

                uindex_t pe_iteration =
                    (i_iteration + i_processing_element / TransFunc::n_subiterations).to_uint();
                uindex_t pe_subiteration =
                    (i_processing_element % TransFunc::n_subiterations).to_uint();
                index_t output_grid_c = input_grid_c - index_t(TransFunc::stencil_radius);
                index_t output_grid_r = input_grid_r - index_t(TransFunc::stencil_radius);
                TDV tdv = tdv_local_state.get_time_dependent_value(i_processing_element /
                                                                   TransFunc::n_subiterations);
                StencilImpl stencil(ID(output_grid_c, output_grid_r), UID(grid_width, grid_height),
                                    pe_iteration, pe_subiteration, tdv,
                                    stencil_buffer[i_processing_element]);

                if (pe_iteration < target_i_iteration) {
                    carry = trans_func(stencil);
                } else {
                    carry = stencil_buffer[i_processing_element][TransFunc::stencil_radius]
                                          [TransFunc::stencil_radius];
                }
            }

            bool is_valid_output =
                input_tile_c >= uindex_1d_t((stencil_diameter - 1) * n_processing_elements);
            is_valid_output &=
                input_tile_r >= uindex_1d_t((stencil_diameter - 1) * n_processing_elements);

            if (is_valid_output) {
                out_pipe::write(carry);
            }

            if (input_tile_r == input_tile_section_height - 1) {
                input_tile_r = 0;
                input_tile_c++;
            } else {
                input_tile_r++;
            }
        }
    }

  private:
    TransFunc trans_func;
    uindex_t i_iteration;
    uindex_t target_i_iteration;
    uindex_t grid_c_offset;
    uindex_t grid_r_offset;
    uindex_t grid_width;
    uindex_t grid_height;
    Cell halo_value;
    TDVKernelArgument tdv_kernel_argument;
};

/**
 * \brief A grid updater that applies an iterative stencil code to a grid.
 *
 * This updater applies an iterative stencil code, defined by the template parameter `F`, to the
 * grid; As often as requested.
 *
 * \tparam F The transition function to apply to input grids.
 *
 * \tparam n_processing_elements (Optimization parameter) The number of processing elements (PEs) to
 * implement. Increasing the number of PEs leads to a higher performance since more iterations are
 * computed in parallel. However, it will also increase the resource and space usage of the design.
 * Too many PEs might also decrease the clock frequency.
 *
 * \tparam tile_width (Optimization parameter) The width of the tile that is updated in one pass.
 * For best hardware utilization, this should be a power of two. Increasing the maximal width of a
 * tile may increase the performance of the design by introducing longer steady-states and reducing
 * halo computation overheads. However, it will also increase the logic resource utilization and
 * might lower the clock frequency.
 *
 * \tparam tile_height (Optimization parameter) The height of the tile that is updated in one pass.
 * Increasing the maximal height of a tile may increase the performance of the design by introducing
 * longer steady-states and reducing halo computation overheads. However, it will also increase the
 * logic and on-chip memory utilization and might lower the clock frequency.
 *
 * \tparam TDVStrategy (Optimization parameter) The precomputation strategy for the time-dependent
 * value system (\ref page-tdv "See guide").
 */
template <concepts::TransitionFunction F, uindex_t n_processing_elements = 1,
          uindex_t tile_width = 1024, uindex_t tile_height = 1024,
          tdv::single_pass::Strategy<F, n_processing_elements> TDVStrategy =
              tdv::single_pass::InlineStrategy>
class StencilUpdate {
  private:
    using Cell = F::Cell;
    using TDVGlobalState = typename TDVStrategy::template GlobalState<F, n_processing_elements>;
    using TDVKernelArgument = typename TDVGlobalState::KernelArgument;

  public:
    /**
     * \brief The radius of an input's tile halo.
     */
    static constexpr uindex_t halo_radius = F::stencil_radius * n_processing_elements;

    /**
     * \brief A shorthand for the used and supported grid type.
     */
    using GridImpl = Grid<Cell, tile_width, tile_height, halo_radius>;

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
        uindex_t iteration_offset = 0;

        /**
         * \brief The number of iterations to compute.
         */
        uindex_t n_iterations = 1;

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
        using in_pipe = sycl::pipe<class tiling_in_pipe, Cell>;
        using out_pipe = sycl::pipe<class tiling_out_pipe, Cell>;
        using ExecutionKernelImpl = StencilUpdateKernel<F, TDVKernelArgument, n_processing_elements,
                                                        tile_width, tile_height, in_pipe, out_pipe>;

        if (params.n_iterations == 0) {
            return GridImpl(source_grid);
        }

        sycl::queue input_kernel_queue =
            sycl::queue(params.device, {sycl::property::queue::in_order{}});
        sycl::queue output_kernel_queue =
            sycl::queue(params.device, {sycl::property::queue::in_order{}});
        sycl::queue working_queue =
            sycl::queue(params.device, {cl::sycl::property::queue::enable_profiling{},
                                        sycl::property::queue::in_order{}});

        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();

        uindex_t iters_per_pass = n_processing_elements / F::n_subiterations;
        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        UID tile_range = source_grid.get_tile_range();
        uindex_t grid_width = source_grid.get_grid_width();
        uindex_t grid_height = source_grid.get_grid_height();

        F trans_func = params.transition_function;
        TDVGlobalState tdv_global_state(trans_func, params.iteration_offset, params.n_iterations);

        auto walltime_start = std::chrono::high_resolution_clock::now();

        uindex_t target_n_iterations = params.iteration_offset + params.n_iterations;
        for (uindex_t i = params.iteration_offset; i < target_n_iterations; i += iters_per_pass) {
            uindex_t iters_in_this_pass = std::min(iters_per_pass, target_n_iterations - i);

            for (uindex_t i_tile_c = 0; i_tile_c < tile_range.c; i_tile_c++) {
                for (uindex_t i_tile_r = 0; i_tile_r < tile_range.r; i_tile_r++) {
                    pass_source->template submit_read<in_pipe>(input_kernel_queue, i_tile_c,
                                                               i_tile_r, params.halo_value);

                    auto work_event = working_queue.submit([&](sycl::handler &cgh) {
                        TDVKernelArgument tdv_kernel_argument(tdv_global_state, cgh, i,
                                                              iters_in_this_pass);
                        uindex_t c_offset = i_tile_c * tile_width;
                        uindex_t r_offset = i_tile_r * tile_height;

                        ExecutionKernelImpl exec_kernel(trans_func, i, target_n_iterations,
                                                        c_offset, r_offset, grid_width, grid_height,
                                                        params.halo_value, tdv_kernel_argument);

                        cgh.single_task<ExecutionKernelImpl>(exec_kernel);
                    });
                    if (params.profiling) {
                        work_events.push_back(work_event);
                    }

                    pass_target->template submit_write<out_pipe>(output_kernel_queue, i_tile_c,
                                                                 i_tile_r);
                }
            }

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
            params.n_iterations * source_grid.get_grid_width() * source_grid.get_grid_height();

        return *pass_source;
    }

    /**
     * \brief Return the accumulated total number of cells processed by this updater.
     *
     * For each call of to \ref operator()(), this is the width times the height of the grid, times
     * the number of computed iterations. This will also be accumulated across multiple calls to
     * \ref operator()().
     */
    uindex_t get_n_processed_cells() const { return n_processed_cells; }

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
    uindex_t n_processed_cells;
    double walltime;
    std::vector<sycl::event> work_events;
};

} // namespace tiling
} // namespace stencil