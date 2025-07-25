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
#include "../../internal/DualIOPipeKernels.hpp"
#include "../../internal/MemoryKernels.hpp"
#include "../../internal/SwitchKernels.hpp"
#include "../Grid.hpp"
#include "StencilUpdateKernel.hpp"
#include <chrono>
#include <type_traits>

namespace stencil {
namespace monotile {
namespace internal {

template <concepts::TransitionFunction F, std::size_t temporal_parallelism = 1,
          std::size_t spatial_parallelism = 1, std::size_t max_grid_height = 1024,
          std::size_t max_grid_width = 1024, std::size_t n_kernels = 1,
          tdv::single_pass::Strategy<F, temporal_parallelism> TDVStrategy =
              tdv::single_pass::InlineStrategy>
class StencilUpdateDesign {
  public:
    using Cell = F::Cell;
    using CellVector = stencil::internal::Padded<std::array<Cell, spatial_parallelism>>;

    template <std::size_t i> class PipeIdentifier;
    using work_in_pipe = sycl::pipe<PipeIdentifier<0>, CellVector>;
    using work_out_pipe = sycl::pipe<PipeIdentifier<n_kernels>, CellVector>;

    using TDV = typename F::TimeDependentValue;
    using TDVGlobalState = TDVStrategy::template GlobalState<F, temporal_parallelism>;
    using TDVKernelArgument = typename TDVGlobalState::KernelArgument;

    /// \brief Shorthand for the used and supported grid type.
    using GridImpl = Grid<typename F::Cell, spatial_parallelism>;

    StencilUpdateDesign(F transition_function, Cell halo_value, std::size_t iteration_offset,
                        std::size_t n_iterations,
                        std::function<int(const sycl::device &)> device_selector)
        : transition_function(transition_function), halo_value(halo_value),
          work_queues(stencil::internal::alloc_queues(device_selector, n_kernels)),
          tdv_global_state(transition_function, iteration_offset, n_iterations) {}

    template <std::size_t i_kernel>
    void submit_work_kernel(std::size_t i_iteration, std::size_t target_i_iteration,
                            sycl::range<2> grid_range)
        requires(i_kernel < n_kernels)
    {
        using in_pipe = sycl::pipe<PipeIdentifier<i_kernel>, CellVector>;
        using out_pipe = sycl::pipe<PipeIdentifier<i_kernel + 1>, CellVector>;
        constexpr size_t local_temporal_parallelism =
            temporal_parallelism / n_kernels +
            ((i_kernel == n_kernels - 1) ? temporal_parallelism % n_kernels : 0);
        using ExecutionKernelImpl =
            internal::StencilUpdateKernel<F, TDVKernelArgument, local_temporal_parallelism,
                                          spatial_parallelism, max_grid_height, max_grid_width,
                                          in_pipe, out_pipe>;

        work_queues[i_kernel].submit([&](sycl::handler &cgh) {
            TDVKernelArgument tdv_kernel_argument(tdv_global_state, cgh, i_iteration,
                                                  local_temporal_parallelism);
            ExecutionKernelImpl exec_kernel(transition_function, i_iteration, target_i_iteration,
                                            grid_range[0], grid_range[1], halo_value,
                                            tdv_kernel_argument);
            cgh.single_task(exec_kernel);
        });

        submit_work_kernel<i_kernel + 1>(i_iteration + local_temporal_parallelism,
                                         target_i_iteration, grid_range);
    }

    template <std::size_t i_kernel>
    void submit_work_kernel(std::size_t i_iteration, std::size_t target_i_iteration,
                            sycl::range<2> grid_range)
        requires(i_kernel == n_kernels)
    {
        return;
    }

    void submit_work_kernels(std::size_t i_iteration, std::size_t target_i_iteration,
                             sycl::range<2> grid_range) {
        submit_work_kernel<0>(i_iteration, target_i_iteration, grid_range);
    }

    void wait_for_queues() {
        for (std::size_t i_queue = 0; i_queue < work_queues.size(); i_queue++) {
            work_queues[i_queue].wait();
        }
    }

  private:
    F transition_function;
    Cell halo_value;

    std::vector<sycl::queue> work_queues;

    TDVGlobalState tdv_global_state;
};

template <concepts::TransitionFunction F, std::size_t temporal_parallelism = 1,
          std::size_t spatial_parallelism = 1, std::size_t max_grid_height = 1024,
          std::size_t max_grid_width = 1024, std::size_t n_kernels = 1,
          tdv::single_pass::Strategy<F, temporal_parallelism> TDVStrategy =
              tdv::single_pass::InlineStrategy>
class LocalStencilUpdateDesign
    : public StencilUpdateDesign<F, temporal_parallelism, spatial_parallelism, max_grid_height,
                                 max_grid_width, n_kernels, TDVStrategy> {
  public:
    using Parent = StencilUpdateDesign<F, temporal_parallelism, spatial_parallelism,
                                       max_grid_height, max_grid_width, n_kernels, TDVStrategy>;
    using Cell = F::Cell;
    using CellVector = stencil::internal::Padded<std::array<Cell, spatial_parallelism>>;

    /// \brief Shorthand for the used and supported grid type.
    using GridImpl = Grid<typename F::Cell, spatial_parallelism>;

    LocalStencilUpdateDesign(F transition_function, Cell halo_value, std::size_t iteration_offset,
                             std::size_t n_iterations,
                             std::function<int(const sycl::device &)> device_selector)
        : Parent(transition_function, halo_value, iteration_offset, n_iterations, device_selector),
          read_queue(), write_queue() {
        std::vector<sycl::queue> queues = stencil::internal::alloc_queues(device_selector, 2);
        read_queue = queues[0];
        write_queue = queues[1];
    }

    GridImpl submit_simulation(GridImpl source_grid, std::size_t iteration_offset,
                               std::size_t n_iterations) {
        using namespace stencil::internal;

        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();

        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        std::size_t target_i_iteration = iteration_offset + n_iterations;
        std::size_t n_passes = int_ceil_div(n_iterations, temporal_parallelism);
        for (std::size_t i_pass = 0; i_pass < n_passes; i_pass++) {
            std::size_t i_iteration = iteration_offset + i_pass * temporal_parallelism;

            submit_pass(*pass_source, *pass_target, i_iteration, target_i_iteration);

            if (i_pass == 0) {
                pass_source = &swap_grid_b;
                pass_target = &swap_grid_a;
            } else {
                std::swap(pass_source, pass_target);
            }
        }

        return *pass_source;
    }

    void submit_pass(GridImpl &pass_source, GridImpl &pass_target, std::size_t i_iteration,
                     std::size_t target_i_iteration) {
        using namespace stencil::internal;
        using in_pipe = typename Parent::work_in_pipe;
        using out_pipe = typename Parent::work_out_pipe;

        using ReadKernel =
            CompleteBufferReadKernel<CellVector, in_pipe, max_grid_height, max_grid_width>;
        read_queue.submit([&](sycl::handler &cgh) {
            ReadKernel kernel(pass_source.get_internal(), cgh);
            cgh.STENCILSTREAM_NAMED_SINGLE_TASK(read_kernel, kernel);
        });

        this->submit_work_kernels(i_iteration, target_i_iteration, pass_source.get_grid_range());

        using WriteKernel =
            CompleteBufferWriteKernel<CellVector, out_pipe, max_grid_height, max_grid_width>;
        write_queue.submit([&](sycl::handler &cgh) {
            WriteKernel kernel(pass_target.get_internal(), cgh);
            cgh.STENCILSTREAM_NAMED_SINGLE_TASK(write_kernel, kernel);
        });
    }

    void wait_for_queues() {
        Parent::wait_for_queues();
        read_queue.wait();
        write_queue.wait();
    }

  private:
    sycl::queue read_queue, write_queue;
};

template <concepts::TransitionFunction F, std::size_t temporal_parallelism = 1,
          std::size_t spatial_parallelism = 1, std::size_t max_grid_height = 1024,
          std::size_t max_grid_width = 1024, std::size_t n_kernels = 1,
          tdv::single_pass::Strategy<F, temporal_parallelism> TDVStrategy =
              tdv::single_pass::InlineStrategy>
class IOPipeStencilUpdateDesign
    : public StencilUpdateDesign<F, temporal_parallelism, spatial_parallelism, max_grid_height,
                                 max_grid_width, n_kernels, TDVStrategy> {
  public:
    using Parent = StencilUpdateDesign<F, temporal_parallelism, spatial_parallelism,
                                       max_grid_height, max_grid_width, n_kernels, TDVStrategy>;
    using Cell = F::Cell;
    using CellVector = stencil::internal::Padded<std::array<Cell, spatial_parallelism>>;

    /// \brief Shorthand for the used and supported grid type.
    using GridImpl = Grid<typename F::Cell, spatial_parallelism>;

    IOPipeStencilUpdateDesign(F transition_function, Cell halo_value, std::size_t iteration_offset,
                              std::size_t n_iterations,
                              std::function<int(const sycl::device &)> device_selector)
        : Parent(transition_function, halo_value, iteration_offset, n_iterations, device_selector),
          read_queue(), recv_queue(), recv_fork_queue(), work_merge_queue(), work_fork_queue(),
          write_merge_queue(), write_queue(), send_queue() {
        std::vector<sycl::queue> queues = stencil::internal::alloc_queues(device_selector, 8);
        read_queue = queues[0];
        recv_queue = queues[1];
        recv_fork_queue = queues[2];
        work_merge_queue = queues[3];
        work_fork_queue = queues[4];
        write_merge_queue = queues[5];
        write_queue = queues[6];
        send_queue = queues[7];
    }

    GridImpl submit_simulation(GridImpl source_grid, std::size_t iteration_offset,
                               std::size_t n_iterations) {
        using namespace stencil::internal;

        int rank, n_ranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();

        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        std::size_t target_i_iteration = iteration_offset + n_iterations;
        std::size_t n_passes = int_ceil_div(n_iterations, temporal_parallelism * n_ranks);
        for (std::size_t i_pass = 0; i_pass < n_passes; i_pass++) {
            std::size_t i_iteration =
                iteration_offset + (i_pass * n_ranks + rank) * temporal_parallelism;

            submit_pass(*pass_source, *pass_target, i_iteration, target_i_iteration, i_pass == 0);

            if (i_pass == 0) {
                pass_source = &swap_grid_b;
                pass_target = &swap_grid_a;
            } else {
                std::swap(pass_source, pass_target);
            }
        }

        return *pass_source;
    }

    void submit_pass(GridImpl &pass_source, GridImpl &pass_target, std::size_t i_iteration,
                     std::size_t target_i_iteration, bool is_root) {
        using namespace stencil::internal;
        using recv_out_pipe = sycl::pipe<class recv_out_pipe_id, CellVector>;
        using recv_to_work_pipe = sycl::pipe<class recv_to_work_pipe_id, CellVector>;
        using recv_to_write_pipe = sycl::pipe<class recv_to_write_pipe_id, CellVector>;
        using read_to_work_pipe = sycl::pipe<class read_to_work_pipe_id, CellVector>;
        using work_to_send_pipe = sycl::pipe<class work_to_send_pipe_id, CellVector>;
        using work_in_pipe = typename Parent::work_in_pipe;
        using work_out_pipe = typename Parent::work_out_pipe;

        sycl::range<2> vect_grid_range = pass_source.get_grid_range(true);
        std::size_t n_vectors = vect_grid_range[0] * vect_grid_range[1];

        using ReadKernel = CompleteBufferReadKernel<CellVector, read_to_work_pipe, max_grid_height,
                                                    max_grid_width>;
        if (is_root) {
            read_queue.submit([&](sycl::handler &cgh) {
                ReadKernel kernel(pass_source.get_internal(), cgh);
                cgh.STENCILSTREAM_NAMED_SINGLE_TASK(read_kernel, kernel);
            });
        }

        using RecvKernel =
            DualIOPipeRecvKernel<CellVector, kernel_input_ch0, kernel_input_ch1, recv_out_pipe>;
        recv_queue.STENCILSTREAM_NAMED_SINGLE_TASK(recv_kernel, RecvKernel(n_vectors));

        using RecvForkKernel =
            ForkSwitchKernel<CellVector, recv_out_pipe, recv_to_work_pipe, recv_to_write_pipe>;
        recv_fork_queue.STENCILSTREAM_NAMED_SINGLE_TASK(recv_fork_kernel,
                                                        RecvForkKernel(n_vectors, !is_root));

        using WorkMergeKernel =
            MergeSwitchKernel<CellVector, recv_to_work_pipe, read_to_work_pipe, work_in_pipe>;
        work_merge_queue.STENCILSTREAM_NAMED_SINGLE_TASK(work_merge_kernel,
                                                         WorkMergeKernel(n_vectors, !is_root));

        this->submit_work_kernels(i_iteration, target_i_iteration, pass_source.get_grid_range());

        using WriteKernel = CompleteBufferWriteKernel<CellVector, recv_to_write_pipe,
                                                      max_grid_height, max_grid_width>;
        if (is_root) {
            write_queue.submit([&](sycl::handler &cgh) {
                cgh.STENCILSTREAM_NAMED_SINGLE_TASK(write_kernel,
                                                    WriteKernel(pass_target.get_internal(), cgh));
            });
        }

        using SendKernel =
            DualIOPipeSendKernel<CellVector, kernel_output_ch2, kernel_output_ch3, work_out_pipe>;
        send_queue.STENCILSTREAM_NAMED_SINGLE_TASK(send_kernel, SendKernel(n_vectors));
    }

    void wait_for_queues() {
        Parent::wait_for_queues();
        read_queue.wait();
        recv_queue.wait();
        recv_fork_queue.wait();
        work_merge_queue.wait();
        work_fork_queue.wait();
        write_merge_queue.wait();
        write_queue.wait();
        send_queue.wait();
    }

  private:
    sycl::queue read_queue, recv_queue, recv_fork_queue, work_merge_queue, work_fork_queue,
        write_merge_queue, write_queue, send_queue;
};

} // namespace internal
} // namespace monotile
} // namespace stencil