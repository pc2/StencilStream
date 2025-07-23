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
#include "../fpga_io/DualIOPipeKernels.hpp"
#include "../fpga_io/MemoryKernels.hpp"
#include "../fpga_io/SwitchKernels.hpp"
#include "Grid.hpp"
#include "StencilUpdateKernel.hpp"
#include <chrono>
#include <type_traits>

namespace stencil {
namespace monotile {

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
template <concepts::TransitionFunction F, std::size_t temporal_parallelism = 1,
          std::size_t spatial_parallelism = 1, std::size_t max_grid_height = 1024,
          std::size_t max_grid_width = 1024, std::size_t n_kernels = 1,
          tdv::single_pass::Strategy<F, temporal_parallelism> TDVStrategy =
              tdv::single_pass::InlineStrategy>
class StencilUpdate {
  private:
    using Cell = F::Cell;
    using CellVector = Padded<std::array<Cell, spatial_parallelism>>;

    template <std::size_t i> class PipeIdentifier;

    using TDV = typename F::TimeDependentValue;
    using TDVGlobalState = TDVStrategy::template GlobalState<F, temporal_parallelism>;
    using TDVKernelArgument = typename TDVGlobalState::KernelArgument;

    struct Queues {
        sycl::queue read_queue, recv_queue, recv_fork_queue, work_merge_queue, work_fork_queue,
            write_merge_queue, write_queue, send_queue;
        std::vector<sycl::queue> work_queues;

        Queues(sycl::device device)
            : read_queue(device, {sycl::property::queue::in_order{}}),
              recv_queue(device, {sycl::property::queue::in_order{}}),
              recv_fork_queue(device, {sycl::property::queue::in_order{}}),
              work_merge_queue(device, {sycl::property::queue::in_order{}}),
              work_fork_queue(device, {sycl::property::queue::in_order{}}),
              write_merge_queue(device, {sycl::property::queue::in_order{}}),
              write_queue(device, {sycl::property::queue::in_order{}}),
              send_queue(device, {sycl::property::queue::in_order{}}), work_queues() {
            for (std::size_t i = 0; i < n_kernels; i++) {
                work_queues.push_back(sycl::queue(device, {sycl::property::queue::in_order{}}));
            }
        }

        void wait() {
            read_queue.wait();
            recv_queue.wait();
            recv_fork_queue.wait();
            work_merge_queue.wait();
            work_fork_queue.wait();
            write_merge_queue.wait();
            write_queue.wait();
            send_queue.wait();
            for (std::size_t i_queue = 0; i_queue < work_queues.size(); i_queue++) {
                work_queues[i_queue].wait();
            }
        }
    };

  public:
    /// \brief Shorthand for the used and supported grid type.
    using GridImpl = Grid<typename F::Cell, spatial_parallelism>;

    enum class OperatingMode {
        SINGLE_DEVICE,
        MPI,
        DEBUG_MPI_ROOT,
        DEBUG_MPI_NONROOT,
    };

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

        OperatingMode operating_mode = OperatingMode::SINGLE_DEVICE;

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
    StencilUpdate(Params params) : params(params), n_processed_cells(0), walltime(0.0) {}

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

#if defined(STENCILSTREAM_TARGET_FPGA_EMU)
        auto device_selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif defined(STENCILSTREAM_TARGET_FPGA)
        auto device_selector = sycl::ext::intel::fpga_selector_v;
#else
    #error                                                                                         \
        "Please link with the `StencilStream_Monotile`, `StencilStream_MonotileEmulator`, or `StencilStream_MonotileReport` targets!"
#endif
        sycl::device device(device_selector);

        Queues queues(device);

        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();

        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        F trans_func = params.transition_function;
        TDVGlobalState tdv_global_state(trans_func, params.iteration_offset, params.n_iterations);

        auto walltime_start = std::chrono::high_resolution_clock::now();

        std::size_t target_i_iteration = params.iteration_offset + params.n_iterations;
        std::size_t n_passes = int_ceil_div(params.n_iterations, temporal_parallelism);
        for (std::size_t i_pass = 0; i_pass < n_passes; i_pass++) {
            std::size_t i_iteration = params.iteration_offset + i_pass * temporal_parallelism;
            submit_pass(queues, *pass_source, *pass_target, tdv_global_state, i_iteration,
                        target_i_iteration);

            if (i_pass == 0) {
                pass_source = &swap_grid_b;
                pass_target = &swap_grid_a;
            } else {
                std::swap(pass_source, pass_target);
            }
        }

        if (params.blocking) {
            queues.wait();
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
    [[deprecated("Not implemented, equal to walltime")]] double get_kernel_runtime() const {
        return walltime;
    }

    /**
     * \brief Return the accumulated runtime of the updater, measured from the host side.
     *
     * For each call to \ref operator()(), the time it took to submit all kernels and, if \ref
     * Params::blocking is true, to finish the computation is recorded and accumulated.
     */
    double get_walltime() const { return walltime; }

  private:
    void submit_pass(Queues &queues, GridImpl &pass_source, GridImpl &pass_target,
                     TDVGlobalState &tdv_global_state, std::size_t i_iteration,
                     std::size_t target_i_iteration) {
        using recv_out_pipe = sycl::pipe<class recv_out_pipe_id, CellVector>;
        using recv_to_work_pipe = sycl::pipe<class recv_to_work_pipe_id, CellVector>;
        using recv_to_write_pipe = sycl::pipe<class recv_to_write_pipe_id, CellVector>;
        using read_to_work_pipe = sycl::pipe<class read_to_work_pipe_id, CellVector>;
        using work_to_send_pipe = sycl::pipe<class work_to_send_pipe_id, CellVector>;
        using work_to_write_pipe = sycl::pipe<class work_to_write_pipe_id, CellVector>;
        using write_in_pipe = sycl::pipe<class write_in_pipe_id, CellVector>;
        using work_in_pipe = sycl::pipe<PipeIdentifier<0>, CellVector>;
        using work_out_pipe = sycl::pipe<PipeIdentifier<n_kernels>, CellVector>;

        sycl::range<2> vect_grid_range = pass_source.get_grid_range(true);
        std::size_t n_vectors = vect_grid_range[0] * vect_grid_range[1];
        bool single_device = params.operating_mode == OperatingMode::SINGLE_DEVICE;
        bool is_root = params.operating_mode ==
                       OperatingMode::DEBUG_MPI_ROOT; // TODO: Also set for actual root.

        using ReadKernel = fpga_io::CompleteBufferReadKernel<CellVector, read_to_work_pipe,
                                                             max_grid_height, max_grid_width>;
        if (single_device || is_root) {
            queues.read_queue.submit([&](sycl::handler &cgh) {
                cgh.single_task(ReadKernel(pass_source.get_internal(), cgh));
            });
        }

        using RecvKernel = fpga_io::DualIOPipeRecvKernel<CellVector, fpga_io::kernel_input_ch0,
                                                         fpga_io::kernel_input_ch1, recv_out_pipe>;
        if (!single_device) {
            queues.recv_queue.single_task(RecvKernel(n_vectors));
        }

        using RecvForkKernel = fpga_io::ForkSwitchKernel<CellVector, recv_out_pipe,
                                                         recv_to_work_pipe, recv_to_write_pipe>;
        if (!single_device) {
            queues.recv_fork_queue.single_task(RecvForkKernel(n_vectors, is_root));
        }

        using WorkMergeKernel = fpga_io::MergeSwitchKernel<CellVector, recv_to_work_pipe,
                                                           read_to_work_pipe, work_in_pipe>;
        queues.work_merge_queue.single_task(WorkMergeKernel(n_vectors, !is_root && !single_device));

        submit_work_kernel<0>(queues.work_queues, tdv_global_state, i_iteration, target_i_iteration,
                              pass_source.get_grid_range());

        using WorkForkKernel = fpga_io::ForkSwitchKernel<CellVector, work_out_pipe,
                                                         work_to_send_pipe, work_to_write_pipe>;
        queues.work_fork_queue.single_task(WorkForkKernel(n_vectors, !single_device));

        using WriteMergeKernel = fpga_io::MergeSwitchKernel<CellVector, work_to_write_pipe,
                                                            recv_to_write_pipe, write_in_pipe>;
        if (single_device || is_root) {
            queues.write_merge_queue.single_task(WriteMergeKernel(n_vectors, single_device));
        }

        using WriteKernel = fpga_io::CompleteBufferWriteKernel<CellVector, write_in_pipe,
                                                               max_grid_height, max_grid_width>;
        if (single_device || is_root) {
            queues.write_queue.submit([&](sycl::handler &cgh) {
                cgh.single_task(WriteKernel(pass_target.get_internal(), cgh));
            });
        }

        using SendKernel =
            fpga_io::DualIOPipeSendKernel<CellVector, fpga_io::kernel_output_ch2,
                                          fpga_io::kernel_output_ch3, work_to_send_pipe>;
        if (!single_device) {
            queues.send_queue.single_task(SendKernel(n_vectors));
        }
    }

    template <std::size_t i_kernel>
    void submit_work_kernel(std::vector<sycl::queue> work_queues, TDVGlobalState &tdv_global_state,
                            std::size_t i_iteration, std::size_t target_i_iteration,
                            sycl::range<2> grid_range)
        requires(i_kernel < n_kernels)
    {
        using in_pipe = sycl::pipe<PipeIdentifier<i_kernel>, CellVector>;
        using out_pipe = sycl::pipe<PipeIdentifier<i_kernel + 1>, CellVector>;
        constexpr size_t local_temporal_parallelism =
            temporal_parallelism / n_kernels +
            ((i_kernel == n_kernels - 1) ? temporal_parallelism % n_kernels : 0);
        using ExecutionKernelImpl =
            StencilUpdateKernel<F, TDVKernelArgument, local_temporal_parallelism,
                                spatial_parallelism, max_grid_height, max_grid_width, in_pipe,
                                out_pipe>;

        work_queues[i_kernel].submit([&](sycl::handler &cgh) {
            TDVKernelArgument tdv_kernel_argument(tdv_global_state, cgh, i_iteration,
                                                  local_temporal_parallelism);
            ExecutionKernelImpl exec_kernel(params.transition_function, i_iteration,
                                            target_i_iteration, grid_range[0], grid_range[1],
                                            params.halo_value, tdv_kernel_argument);
            cgh.single_task(exec_kernel);
        });

        submit_work_kernel<i_kernel + 1>(work_queues, tdv_global_state,
                                         i_iteration + local_temporal_parallelism,
                                         target_i_iteration, grid_range);
    }

    template <std::size_t i_kernel>
    void submit_work_kernel(std::vector<sycl::queue> work_queues, TDVGlobalState &tdv_global_state,
                            std::size_t i_iteration, std::size_t target_i_iteration,
                            sycl::range<2> grid_range)
        requires(i_kernel == n_kernels)
    {
        return;
    }

    Params params;
    std::size_t n_processed_cells;
    double walltime;
};

} // namespace monotile
} // namespace stencil