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
#include "SingleContextExecutor.hpp"
#include "tiling/ExecutionKernel.hpp"
#include "tiling/Grid.hpp"

namespace stencil {
/**
 * \brief The default stencil executor.
 *
 * Unlike \ref MonotileExecutor, this executors supports any grid range by tiling the grid as
 * described in \ref tiling, at the cost of complexer IO and a computational overhead for every
 * tile.
 *
 * This executor is called `StencilExecutor` since was the first one in StencilStream 2.x. A better
 * name for it would be `TilingExecutor`.
 *
 * \tparam T The cell type.
 * \tparam stencil_radius The radius of the stencil buffer supplied to the transition function.
 * \tparam TransFunc The type of the transition function.
 * \tparam pipeline_length The number of hardware execution stages per kernel. Must be at least 1.
 * Defaults to 1.
 * \tparam tile_width The number of columns in a tile and maximum number of columns in a grid.
 * Defaults to 1024.
 * \tparam tile_height The number of rows in a tile and maximum number of rows in a grid. Defaults
 * to 1024.
 */
template <typename T, uindex_t stencil_radius, typename TransFunc, uindex_t pipeline_length = 1,
          uindex_t tile_width = 1024, uindex_t tile_height = 1024>
class StencilExecutor : public SingleContextExecutor<T, stencil_radius, TransFunc> {
  public:
    /**
     * \brief The number of cells that have be added to the tile in every direction to form the
     * complete input.
     */
    static constexpr uindex_t halo_radius = stencil_radius * pipeline_length;

    /**
     * \brief Shorthand for the parent class.
     */
    using Parent = SingleContextExecutor<T, stencil_radius, TransFunc>;

    /**
     * \brief Create a new stencil executor.
     *
     * \param halo_value The value of cells in the grid halo.
     * \param trans_func An instance of the transition function type.
     */
    StencilExecutor(T halo_value, TransFunc trans_func)
        : Parent(halo_value, trans_func),
          input_grid(cl::sycl::buffer<T, 2>(cl::sycl::range<2>(0, 0))) {}

    void set_input(cl::sycl::buffer<T, 2> input_buffer) override {
        this->input_grid = GridImpl(input_buffer);
    }

    void copy_output(cl::sycl::buffer<T, 2> output_buffer) override {
        input_grid.copy_to(output_buffer);
    }

    UID get_grid_range() const override { return input_grid.get_grid_range(); }

    void run(uindex_t n_generations) override {
        using in_pipe = cl::sycl::pipe<class tiling_in_pipe, T>;
        using out_pipe = cl::sycl::pipe<class tiling_out_pipe, T>;
        using ExecutionKernelImpl =
            tiling::ExecutionKernel<TransFunc, T, stencil_radius, pipeline_length, tile_width,
                                    tile_height, in_pipe, out_pipe>;

        cl::sycl::queue input_queue = this->new_queue(true);
        cl::sycl::queue work_queue = this->new_queue(true);
        cl::sycl::queue output_queue = this->new_queue(true);

        uindex_t target_i_generation = this->get_i_generation() + n_generations;
        uindex_t grid_width = input_grid.get_grid_range().c;
        uindex_t grid_height = input_grid.get_grid_range().r;

        while (this->get_i_generation() < target_i_generation) {
            GridImpl output_grid = input_grid.make_output_grid();

            std::vector<cl::sycl::event> events;
            events.reserve(input_grid.get_tile_range().c * input_grid.get_tile_range().r);

            for (uindex_t c = 0; c < input_grid.get_tile_range().c; c++) {
                for (uindex_t r = 0; r < input_grid.get_tile_range().r; r++) {
                    input_grid.template submit_tile_input<in_pipe>(input_queue, UID(c, r));

                    cl::sycl::event computation_event = work_queue.submit([&](cl::sycl::handler &cgh) {
                        cgh.single_task<class TilingExecutionKernel>(ExecutionKernelImpl(
                            this->get_trans_func(), this->get_i_generation(), target_i_generation,
                            c * tile_width, r * tile_height, grid_width, grid_height,
                            this->get_halo_value()));
                    });
                    events.push_back(computation_event);

                    output_grid.template submit_tile_output<out_pipe>(output_queue, UID(c, r));
                }
            }

            input_grid = output_grid;

            double earliest_start = std::numeric_limits<double>::max();
            double latest_end = std::numeric_limits<double>::min();

            for (cl::sycl::event event : events) {
                earliest_start = std::min(earliest_start, RuntimeSample::start_of_event(event));
                latest_end = std::max(latest_end, RuntimeSample::end_of_event(event));
            }
            this->get_runtime_sample().add_pass(latest_end - earliest_start);

            this->inc_i_generation(
                std::min(target_i_generation - this->get_i_generation(), pipeline_length));
        }
    }

  private:
    using GridImpl = tiling::Grid<T, tile_width, tile_height, halo_radius>;
    GridImpl input_grid;
};
} // namespace stencil