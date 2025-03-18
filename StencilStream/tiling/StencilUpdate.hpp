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
#include "../tdv/SinglePassStrategies.hpp"
#include "Grid.hpp"
#include "StencilUpdateKernel.hpp"

#include <chrono>
#include <list>

namespace stencil {
namespace tiling {

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
          std::size_t tile_width = 1024, std::size_t n_kernels = 1,
          tdv::single_pass::Strategy<F, temporal_parallelism> TDVStrategy =
              tdv::single_pass::InlineStrategy>
class StencilUpdate {
  private:
    using Cell = F::Cell;

    template <std::size_t i> class PipeIdentifier;

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
        using in_pipe = sycl::pipe<PipeIdentifier<0>, std::array<Cell, spatial_parallelism>>;
        using out_pipe =
            sycl::pipe<PipeIdentifier<n_kernels>, std::array<Cell, spatial_parallelism>>;

        // Never submitted, but necessary to compute total halo height an width.
        using FullExecutionKernelImpl =
            StencilUpdateKernel<F, TDVKernelArgument, temporal_parallelism, 0, spatial_parallelism,
                                tile_height, tile_width, in_pipe, out_pipe>;
        constexpr std::size_t halo_height = FullExecutionKernelImpl::get_halo_height();
        constexpr std::size_t halo_width = FullExecutionKernelImpl::get_halo_width();

        if (params.n_iterations == 0) {
            return GridImpl(source_grid);
        }

        sycl::queue queue = sycl::queue(params.device, {sycl::property::queue::enable_profiling{}});

        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();

        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        sycl::range<2> tile_range = source_grid.get_tile_range(tile_height, tile_width);
        sycl::range<2> grid_range = source_grid.get_grid_range();

        TDVGlobalState tdv_global_state(params.transition_function, params.iteration_offset,
                                        params.n_iterations);

        auto walltime_start = std::chrono::high_resolution_clock::now();

        std::size_t target_i_iteration = params.iteration_offset + params.n_iterations;
        for (std::size_t i = params.iteration_offset; i < target_i_iteration;
             i += temporal_parallelism) {
            for (std::size_t i_tile_r = 0; i_tile_r < tile_range[0]; i_tile_r++) {
                for (std::size_t i_tile_c = 0; i_tile_c < tile_range[1]; i_tile_c++) {
                    sycl::id<2> i_tile(i_tile_r, i_tile_c);

                    pass_source->template submit_read<in_pipe, tile_height, tile_width, halo_height,
                                                      halo_width>(queue, i_tile_r, i_tile_c,
                                                                  params.halo_value);

                    std::list<sycl::event> pass_work_events = submit_work_kernel<0>(
                        queue, tdv_global_state, i, target_i_iteration, grid_range, i_tile);
                    if (params.profiling) {
                        for (sycl::event event : pass_work_events) {
                            work_events.push_back(event);
                        };
                    }

                    pass_target->template submit_write<out_pipe, tile_height, tile_width>(
                        queue, i_tile_r, i_tile_c);
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
            queue.wait();
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

    /**
     * \brief Return the accumulated runtime of the updater, measured from the host side.
     *
     * For each call to \ref operator()(), the time it took to submit all kernels and, if \ref
     * Params::blocking is true, to finish the computation is recorded and accumulated.
     */
    double get_walltime() const { return walltime; }

  private:
    template <std::size_t i_kernel>
    std::list<sycl::event> submit_work_kernel(sycl::queue queue, TDVGlobalState &tdv_global_state,
                                              std::size_t i_iteration,
                                              std::size_t target_i_iteration,
                                              sycl::range<2> grid_range, sycl::id<2> i_tile)
        requires(i_kernel < n_kernels)
    {
        using in_pipe = sycl::pipe<PipeIdentifier<i_kernel>, std::array<Cell, spatial_parallelism>>;
        using out_pipe =
            sycl::pipe<PipeIdentifier<i_kernel + 1>, std::array<Cell, spatial_parallelism>>;

        constexpr size_t local_temporal_parallelism =
            temporal_parallelism / n_kernels +
            ((i_kernel == n_kernels - 1) ? temporal_parallelism % n_kernels : 0);
        constexpr size_t remaining_temporal_parallelism =
            temporal_parallelism - local_temporal_parallelism -
            i_kernel * (temporal_parallelism / n_kernels);

        using ExecutionKernelImpl =
            StencilUpdateKernel<F, TDVKernelArgument, local_temporal_parallelism,
                                remaining_temporal_parallelism, spatial_parallelism, tile_height,
                                tile_width, in_pipe, out_pipe>;

        sycl::event work_event = queue.submit([&](sycl::handler &cgh) {
            TDVKernelArgument tdv_kernel_argument(tdv_global_state, cgh, i_iteration,
                                                  local_temporal_parallelism);
            std::ptrdiff_t r_offset = i_tile[0] * tile_height;
            std::ptrdiff_t c_offset = i_tile[1] * tile_width;

            ExecutionKernelImpl exec_kernel(params.transition_function, i_iteration,
                                            target_i_iteration, r_offset, c_offset, grid_range[0],
                                            grid_range[1], params.halo_value, tdv_kernel_argument);

            cgh.single_task<ExecutionKernelImpl>(exec_kernel);
        });

        std::list<sycl::event> following_events = submit_work_kernel<i_kernel + 1>(
            queue, tdv_global_state, i_iteration + local_temporal_parallelism, target_i_iteration,
            grid_range, i_tile);
        following_events.push_front(work_event);
        return following_events;
    }

    template <std::size_t i_kernel>
    std::list<sycl::event> submit_work_kernel(sycl::queue queue, TDVGlobalState &tdv_global_state,
                                              std::size_t i_iteration,
                                              std::size_t target_i_iteration,
                                              sycl::range<2> grid_range, sycl::id<2> i_tile)
        requires(i_kernel == n_kernels)
    {
        return std::list<sycl::event>();
    }

    Params params;
    std::size_t n_processed_cells;
    double walltime;
    std::vector<sycl::event> work_events;
};

} // namespace tiling
} // namespace stencil