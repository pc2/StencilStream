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
 * \tparam temporal_parallelism The number of iterations to compute in parallel. Notice that
 * subiterations within one iteration are always computed in parallel.
 *
 * \tparam spatial_parallelsim The number of cells to update in parallel within one iteration.
 *
 * \tparam output_tile_height The number of rows in a grid tile.
 *
 * \tparam output_tile_width The number of columns in a grid tile.
 *
 * \tparam in_pipe The pipe to read from.
 *
 * \tparam out_pipe The pipe to write to.
 */
template <concepts::TransitionFunction TransFunc,
          tdv::single_pass::KernelArgument<TransFunc> TDVKernelArgument,
          std::size_t temporal_parallelism, std::size_t spatial_parallelism,
          std::size_t output_tile_height, std::size_t output_tile_width, typename in_pipe,
          typename out_pipe>
class StencilUpdateKernel {
  private:
    using Cell = typename TransFunc::Cell;
    using CellVector = std::array<Cell, spatial_parallelism>;
    using TDV = typename TransFunc::TimeDependentValue;
    using StencilImpl = Stencil<Cell, TransFunc::stencil_radius, TDV>;
    using TDVLocalState = typename TDVKernelArgument::LocalState;

    static constexpr std::size_t n_processing_elements =
        temporal_parallelism * TransFunc::n_subiterations;

    static constexpr std::size_t stencil_radius = TransFunc::stencil_radius;
    static constexpr std::size_t vect_stencil_buffer_lead =
        int_ceil_div(stencil_radius, spatial_parallelism);
    static constexpr std::size_t stencil_buffer_lead =
        vect_stencil_buffer_lead * spatial_parallelism;
    static constexpr std::size_t stencil_buffer_height = 2 * stencil_radius + 1;
    static constexpr std::size_t stencil_buffer_width =
        stencil_radius + spatial_parallelism + stencil_buffer_lead;

    static constexpr std::size_t halo_height = stencil_radius * n_processing_elements;
    static constexpr std::size_t halo_width = stencil_buffer_lead * n_processing_elements;
    static constexpr std::size_t vect_halo_width = vect_stencil_buffer_lead * n_processing_elements;

    static constexpr std::size_t vect_output_tile_width = output_tile_width / spatial_parallelism;
    static_assert(output_tile_width % spatial_parallelism == 0);

    static constexpr std::size_t input_tile_height = 2 * halo_height + output_tile_height;
    static constexpr std::size_t vect_input_tile_width =
        2 * vect_halo_width + vect_output_tile_width;
    static constexpr std::size_t input_tile_width = vect_input_tile_width * spatial_parallelism;

    static constexpr unsigned long bits_1d =
        std::bit_width(std::max(input_tile_height, input_tile_width));
    using index_1d_t = ac_int<bits_1d + 1, true>;
    using uindex_1d_t = ac_int<bits_1d, false>;

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
     * \param grid_r_offset The row offset of the processed tile relative to the grid's origin,
     * not including the halo. For example, for the most north-western tile the offset will always
     * be (0,0), not (-halo_radius,-halo_radius)
     *
     * \param grid_c_offset The column offset of the processed tile relative to the grid's origin.
     * See `grid_r_offset` for details.
     *
     * \param grid_height The number of cell rows in the grid.
     *
     * \param grid_width The number of cell columns in the grid.
     *
     * \param halo_value The value of cells in the grid halo.
     *
     * \param tdv_kernel_argument The argument for the TDV system that is passed from the host to
     * the device. This may for example contain global memory accessors.
     */
    StencilUpdateKernel(TransFunc trans_func, std::size_t i_iteration,
                        std::size_t target_i_iteration, std::size_t grid_r_offset,
                        std::size_t grid_c_offset, std::size_t grid_height, std::size_t grid_width,
                        Cell halo_value, TDVKernelArgument tdv_kernel_argument)
        : trans_func(trans_func), i_iteration(i_iteration), target_i_iteration(target_i_iteration),
          grid_r_offset(grid_r_offset), grid_c_offset(grid_c_offset), grid_height(grid_height),
          grid_width(grid_width), halo_value(halo_value), tdv_kernel_argument(tdv_kernel_argument) {
        assert(grid_r_offset % output_tile_height == 0);
        assert(grid_c_offset % output_tile_width == 0);
    }

    static constexpr std::size_t get_halo_height() { return halo_height; }

    static constexpr std::size_t get_halo_width() { return halo_width; }

    /**
     * \brief Execute the configured operations.
     */
    void operator()() const {
        TDVLocalState tdv_local_state(tdv_kernel_argument);

        /*
         * The intel::numbanks attribute requires a power of two as it's argument and if the
         * number of processing elements isn't a power of two, it would produce an error. Therefore,
         * we calculate the next power of two and use it to allocate the cache. The compiler is
         * smart enough to see that these additional banks in the cache aren't used and therefore
         * optimizes them away.
         */
        [[intel::fpga_memory,
          intel::numbanks(2 * std::bit_ceil(n_processing_elements))]] Padded<CellVector>
            cache[2][vect_input_tile_width][std::bit_ceil(n_processing_elements)]
                 [stencil_buffer_height - 1];
        [[intel::fpga_register]] Cell stencil_buffer[n_processing_elements][stencil_buffer_height]
                                                    [stencil_buffer_width];

        uindex_1d_t output_tile_section_height =
            std::min(output_tile_height, grid_height - grid_r_offset);
        uindex_1d_t output_tile_section_width =
            std::min(output_tile_width, grid_width - grid_c_offset);
        uindex_1d_t vect_output_tile_section_width =
            int_ceil_div<uindex_1d_t>(output_tile_section_width, spatial_parallelism);

        uindex_1d_t input_tile_section_height = output_tile_section_height + 2 * halo_height;
        uindex_1d_t vect_input_tile_section_width =
            vect_output_tile_section_width + 2 * vect_halo_width;

        [[intel::loop_coalesce(2)]] for (index_1d_t input_tile_r = 0;
                                         input_tile_r < input_tile_section_height; input_tile_r++) {
            for (index_1d_t vect_input_tile_c = 0;
                 vect_input_tile_c < vect_input_tile_section_width; vect_input_tile_c++) {
                [[intel::fpga_register]] CellVector carry = in_pipe::read();

#pragma unroll
                for (std::size_t i_processing_element = 0;
                     i_processing_element < n_processing_elements; i_processing_element++) {

                    index_1d_t rel_input_grid_r =
                        index_1d_t(input_tile_r) - index_1d_t(halo_height) -
                        index_1d_t(i_processing_element * TransFunc::stencil_radius);
                    std::size_t input_grid_r = grid_r_offset + std::ptrdiff_t(rel_input_grid_r);

                    uindex_1d_t input_tile_c = vect_input_tile_c * spatial_parallelism;
                    index_1d_t rel_input_grid_c =
                        index_1d_t(input_tile_c) - index_1d_t(halo_width) -
                        index_1d_t(i_processing_element * stencil_buffer_lead);
                    std::size_t input_grid_c = grid_c_offset + std::ptrdiff_t(rel_input_grid_c);

                    /*
                     * Shift every value in the stencil_buffer left.
                     * This operation does not touch the values in the right-most column, which will
                     * be filled from the cache and the new input value later.
                     */
#pragma unroll
                    for (std::size_t r = 0; r < stencil_buffer_height; r++) {
#pragma unroll
                        for (std::size_t c = 0; c < stencil_buffer_width - spatial_parallelism;
                             c++) {
                            stencil_buffer[i_processing_element][r][c] =
                                stencil_buffer[i_processing_element][r][c + spatial_parallelism];
                        }
                    }

                    // Update the stencil buffer and cache with previous cache contents and the new
                    // input cell.
#pragma unroll
                    for (std::size_t cache_r = 0; cache_r < stencil_buffer_height; cache_r++) {
                        CellVector new_vector;
                        if (cache_r == stencil_buffer_height - 1) {
                            // grid_*_offset is unsigned, can not directly test for negativity.
                            bool is_halo_row = (grid_r_offset == 0 && rel_input_grid_r < 0) ||
                                               input_grid_r >= grid_height;

#pragma unroll
                            for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                                bool is_halo_cell =
                                    is_halo_row ||
                                    (grid_c_offset == 0 && (rel_input_grid_c + i_cell) < 0) ||
                                    (input_grid_c + i_cell >= grid_width);
                                if (is_halo_cell) {
                                    new_vector[i_cell] = halo_value;
                                } else {
                                    new_vector[i_cell] = carry[i_cell];
                                }
                            }
                        } else {
                            new_vector = cache[input_tile_r[0]][vect_input_tile_c]
                                              [i_processing_element][cache_r]
                                                  .value;
                        }

#pragma unroll
                        for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                            stencil_buffer[i_processing_element][cache_r]
                                          [stencil_buffer_width - spatial_parallelism + i_cell] =
                                              new_vector[i_cell];
                        }

                        if (cache_r > 0) {
                            cache[(~input_tile_r)[0]][vect_input_tile_c][i_processing_element]
                                 [cache_r - 1]
                                     .value = new_vector;
                        }
                    }

                    std::size_t pe_iteration =
                        i_iteration +
                        std::size_t(i_processing_element / TransFunc::n_subiterations);
                    std::size_t pe_subiteration = i_processing_element % TransFunc::n_subiterations;
                    std::size_t output_grid_r = input_grid_r - TransFunc::stencil_radius;
                    TDV tdv = tdv_local_state.get_time_dependent_value(i_processing_element /
                                                                       TransFunc::n_subiterations);

#pragma unroll
                    for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                        std::size_t output_grid_c = input_grid_c - stencil_buffer_lead + i_cell;
                        StencilImpl stencil(sycl::id<2>(output_grid_r, output_grid_c),
                                            sycl::range<2>(grid_height, grid_width), pe_iteration,
                                            pe_subiteration, tdv);

#pragma unroll
                        for (std::size_t stencil_r = 0; stencil_r < 2 * stencil_radius + 1;
                             stencil_r++) {
#pragma unroll
                            for (std::size_t stencil_c = 0; stencil_c < 2 * stencil_radius + 1;
                                 stencil_c++) {
                                stencil[sycl::id<2>(stencil_r, stencil_c)] =
                                    stencil_buffer[i_processing_element][stencil_r]
                                                  [i_cell + stencil_c];
                            }
                        }

                        if (output_grid_c < grid_width) {
                            if (pe_iteration < target_i_iteration) {
                                carry[i_cell] = trans_func(stencil);
                            } else {
                                carry[i_cell] = stencil_buffer[i_processing_element][stencil_radius]
                                                              [stencil_radius + i_cell];
                            }
                        } else {
                            carry[i_cell] = halo_value;
                        }
                    }
                }

                bool is_valid_output = (input_tile_r >= uindex_1d_t(2 * halo_height)) &&
                                       (vect_input_tile_c >= uindex_1d_t(2 * vect_halo_width));

                if (is_valid_output) {
                    out_pipe::write(carry);
                }
            }
        }
    }

  private:
    TransFunc trans_func;
    std::size_t i_iteration;
    std::size_t target_i_iteration;
    std::size_t grid_r_offset;
    std::size_t grid_c_offset;
    std::size_t grid_height;
    std::size_t grid_width;
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
 * \tparam temporal_parallelism (Optimization parameter) The number of iterations to compute in
 * parallel. Increasing this parameter leads to a higher performance, but it will also increase the
 * resource and space usage of the design. Excessive values may also decrease the clock frequency.
 * Also notice that subiterations within one iteration are always computed in parallel.
 *
 * \tparam spatial_parallelism (Optimization parameter) The number of cells to update in parallel
 * within one iteration. Increasing this parameter leads to a higher performance as long as the
 * product of the parameter and the size of the cell is below the physical memory word width. Common
 * values are 512 bits or 64 bytes.
 *
 * \tparam tile_height (Optimization parameter) The height of the tile that is updated in one pass.
 * For best hardware utilization, this should be a power of two. Increasing the maximal height of a
 * tile may increase the performance of the design by introducing longer steady-states and reducing
 * halo computation overheads. However, it will also increase the logic resource utilization and
 * might lower the clock frequency.
 *
 * \tparam tile_width (Optimization parameter) The width of the tile that is updated in one pass.
 * Increasing the maximal width of a tile may increase the performance of the design by introducing
 * longer steady-states and reducing halo computation overheads. However, it will also increase the
 * logic and on-chip memory utilization and might lower the clock frequency.
 *
 * \tparam TDVStrategy (Optimization parameter) The precomputation strategy for the time-dependent
 * value system (\ref page-tdv "See guide").
 */
template <concepts::TransitionFunction F, std::size_t temporal_parallelism = 1,
          std::size_t spatial_parallelism = 1, std::size_t tile_height = 1024,
          std::size_t tile_width = 1024,
          tdv::single_pass::Strategy<F, temporal_parallelism> TDVStrategy =
              tdv::single_pass::InlineStrategy>
class StencilUpdate {
  private:
    using Cell = F::Cell;
    using TDVGlobalState = typename TDVStrategy::template GlobalState<F, temporal_parallelism>;
    using TDVKernelArgument = typename TDVGlobalState::KernelArgument;

  public:
    /**
     * \brief A shorthand for the used and supported grid type.
     */
    using GridImpl = Grid<Cell, spatial_parallelism>;

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
        using in_pipe = sycl::pipe<class tiling_in_pipe, std::array<Cell, spatial_parallelism>>;
        using out_pipe = sycl::pipe<class tiling_out_pipe, std::array<Cell, spatial_parallelism>>;
        using ExecutionKernelImpl =
            StencilUpdateKernel<F, TDVKernelArgument, temporal_parallelism, spatial_parallelism,
                                tile_height, tile_width, in_pipe, out_pipe>;
        constexpr std::size_t halo_height = ExecutionKernelImpl::get_halo_height();
        constexpr std::size_t halo_width = ExecutionKernelImpl::get_halo_width();

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

        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        sycl::range<2> tile_range = source_grid.get_tile_range(tile_height, tile_width);
        std::size_t grid_height = source_grid.get_grid_height();
        std::size_t grid_width = source_grid.get_grid_width();

        F trans_func = params.transition_function;
        TDVGlobalState tdv_global_state(trans_func, params.iteration_offset, params.n_iterations);

        auto walltime_start = std::chrono::high_resolution_clock::now();

        std::size_t target_n_iterations = params.iteration_offset + params.n_iterations;
        for (std::size_t i = params.iteration_offset; i < target_n_iterations;
             i += temporal_parallelism) {
            std::size_t iters_in_this_pass =
                std::min(temporal_parallelism, target_n_iterations - i);

            for (std::size_t i_tile_r = 0; i_tile_r < tile_range[0]; i_tile_r++) {
                for (std::size_t i_tile_c = 0; i_tile_c < tile_range[1]; i_tile_c++) {
                    pass_source->template submit_read<in_pipe, tile_height, tile_width, halo_height,
                                                      halo_width>(input_kernel_queue, i_tile_r,
                                                                  i_tile_c, params.halo_value);

                    auto work_event = working_queue.submit([&](sycl::handler &cgh) {
                        TDVKernelArgument tdv_kernel_argument(tdv_global_state, cgh, i,
                                                              iters_in_this_pass);
                        std::size_t r_offset = i_tile_r * tile_height;
                        std::size_t c_offset = i_tile_c * tile_width;

                        ExecutionKernelImpl exec_kernel(trans_func, i, target_n_iterations,
                                                        r_offset, c_offset, grid_height, grid_width,
                                                        params.halo_value, tdv_kernel_argument);

                        cgh.single_task<ExecutionKernelImpl>(exec_kernel);
                    });
                    if (params.profiling) {
                        work_events.push_back(work_event);
                    }

                    pass_target->template submit_write<out_pipe, tile_height, tile_width>(
                        output_kernel_queue, i_tile_r, i_tile_c);
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

} // namespace tiling
} // namespace stencil