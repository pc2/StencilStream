/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "SingleQueueExecutor.hpp"
#include "tiling/ExecutionKernel.hpp"
#include "tiling/Grid.hpp"

namespace stencil {
/**
 * \brief Execution coordinator.
 *
 * The `StencilExecutor` binds the different parts of StencilStream together and provides a unified
 * interface for applications. Users create a stencil executor, configure it and then use it to run
 * the payload computations. It has multiple logical attributes that can be configured:
 *
 * ### Grid
 *
 * The grid is the logical array of cells, set with \ref StencilExecutor.set_input A stencil
 * executor does not work in place and a buffer used to initialize the grid can be used for other
 * tasks afterwards. The \ref StencilExecutor.run method alters the state of the grid and the grid
 * can be copied back to a given buffer using \ref StencilExecutor.copy_output.
 *
 * ### SYCL queue
 *
 * The queue provides the OpenCL platform, device, context and queue to execute the kernels. A
 * stencil executor does not have a queue when constructed and \ref StencilExecutor.run tries to
 * select the FPGA emulation device if no queue has been configured yet. \ref
 * StencilExecutor.set_queue is used to directly set the queue, but \ref
 * StencilExecutor.select_emulator and \ref StencilExecutor.select_fpga can be used to automatically
 * configure the FPGA emulator or the FPGA, respectively.
 *
 * ### Transition Function
 *
 * A stencil executor stores an instance of the transition function since it may require some
 * configuration and runtime-dynamic parameters too. An instance is required for the initialization,
 * but it may be replaced at any time with \ref StencilExecutor.set_trans_func.
 *
 * ### Generation Index
 *
 * This is the generation index of the current state of the grid. \ref StencilExecutor.run updates
 * and therefore, it can be ignored in most instances. However, it can be reset if a transition
 * function needs it.
 *
 * \tparam T Cell value type.
 * \tparam stencil_radius The static, maximal Chebyshev distance of cells in a stencil to the
 * central cell. Must be at least 1. \tparam TransFunc An invocable type that maps a \ref Stencil to
 * the next generation of the stencil's central cell. \tparam pipeline_length The number of hardware
 * execution stages. Must be at least 1. Defaults to 1. \tparam tile_width The number of columns in
 * a tile. Defaults to 1024. \tparam tile_height The number of rows in a tile. Defaults to 1024.
 * \tparam burst_size The number of bytes to load/store in one burst. Defaults to 1024.
 */
template <typename T, uindex_t stencil_radius, typename TransFunc, uindex_t pipeline_length = 1,
          uindex_t tile_width = 1024, uindex_t tile_height = 1024, uindex_t burst_size = 1024>
class StencilExecutor : public SingleQueueExecutor<T, stencil_radius, TransFunc> {
  public:
    static constexpr uindex_t burst_length = std::min<uindex_t>(1, burst_size / sizeof(T));
    static constexpr uindex_t halo_radius = stencil_radius * pipeline_length;
    using Parent = SingleQueueExecutor<T, stencil_radius, TransFunc>;

    /**
     * \brief Create a new stencil executor.
     *
     * \param halo_value The value of cells in the grid halo.
     * \param trans_func An instance of the transition function type.
     */
    StencilExecutor(T halo_value, TransFunc trans_func)
        : Parent(halo_value, trans_func),
          input_grid(cl::sycl::buffer<T, 2>(cl::sycl::range<2>(0, 0))) {}

    /**
     * \brief Set the state of the grid.
     *
     * \param input_buffer A buffer containing the new state of the grid.
     */
    void set_input(cl::sycl::buffer<T, 2> input_buffer) override {
        this->input_grid = GridImpl(input_buffer);
    }

    /**
     * \brief Copy the current state of the grid to the buffer.
     *
     * The `output_buffer` has to have the exact range as returned by \ref
     * StencilExecutor.get_grid_range.
     *
     * \param output_buffer Copy the state of the grid to this buffer.
     */
    void copy_output(cl::sycl::buffer<T, 2> output_buffer) override {
        input_grid.copy_to(output_buffer);
    }

    /**
     * \brief Return the range of the grid.
     *
     * \return The range of the grid.
     */
    UID get_grid_range() const override { return input_grid.get_grid_range(); }

    void run(uindex_t n_generations) override {
        using in_pipe = cl::sycl::pipe<class tiling_in_pipe, T>;
        using out_pipe = cl::sycl::pipe<class tiling_out_pipe, T>;
        using ExecutionKernelImpl =
            tiling::ExecutionKernel<TransFunc, T, stencil_radius, pipeline_length, tile_width,
                                    tile_height, in_pipe, out_pipe>;

        cl::sycl::queue &queue = this->get_queue();

        uindex_t target_i_generation = this->get_i_generation() + n_generations;
        uindex_t grid_width = input_grid.get_grid_range().c;
        uindex_t grid_height = input_grid.get_grid_range().r;

        while (this->get_i_generation() < target_i_generation) {
            GridImpl output_grid = input_grid.make_output_grid();

            std::vector<cl::sycl::event> events;
            events.reserve(input_grid.get_tile_range().c * input_grid.get_tile_range().r);

            for (uindex_t c = 0; c < input_grid.get_tile_range().c; c++) {
                for (uindex_t r = 0; r < input_grid.get_tile_range().r; r++) {
                    input_grid.template submit_tile_input<in_pipe>(queue, UID(c, r));

                    cl::sycl::event computation_event = queue.submit([&](cl::sycl::handler &cgh) {
                        cgh.single_task(ExecutionKernelImpl(
                            this->get_trans_func(), this->get_i_generation(), target_i_generation,
                            c * tile_width, r * tile_height, grid_width, grid_height,
                            this->get_halo_value()));
                    });
                    events.push_back(computation_event);

                    output_grid.template submit_tile_output<out_pipe>(queue, UID(c, r));
                }
            }

            input_grid = output_grid;

            if (this->is_runtime_analysis_enabled()) {
                double earliest_start = std::numeric_limits<double>::max();
                double latest_end = std::numeric_limits<double>::min();

                for (cl::sycl::event event : events) {
                    earliest_start = std::min(earliest_start, RuntimeSample::start_of_event(event));
                    latest_end = std::max(latest_end, RuntimeSample::end_of_event(event));
                }
                this->get_runtime_sample().add_pass(latest_end - earliest_start);
            }

            this->inc_i_generation(
                std::min(target_i_generation - this->get_i_generation(), pipeline_length));
        }
    }

  private:
    using GridImpl = tiling::Grid<T, tile_width, tile_height, halo_radius, burst_length>;
    GridImpl input_grid;
};
} // namespace stencil