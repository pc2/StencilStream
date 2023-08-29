/*
 * Copyright © 2020-2023 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "tiling/TiledGrid.hpp"

#include <algorithm>

namespace stencil {

/**
 * \brief The default stencil executor.
 *
 * Unlike \ref MonotileExecutor, this executors supports any grid range by tiling the grid as
 * described in \ref tiling, at the cost of complexer IO and a computational overhead for every
 * tile.
 *
 * This executor is called `TilingExecutor` since was the first one in StencilStream 2.x. A better
 * name for it would be `TilingExecutor`.
 *
 * \tparam TransFunc The type of the transition function.
 * \tparam n_processing_elements The number of processing elements per kernel. Must be at least 1.
 * Defaults to 1.
 * \tparam tile_width The number of columns in a tile and maximum number of columns in a grid.
 * Defaults to 1024.
 * \tparam tile_height The number of rows in a tile and maximum number of rows in a grid. Defaults
 * to 1024.
 */
template <TransitionFunction TransFunc, tdv::HostState TDVS, uindex_t n_processing_elements = 1,
          uindex_t tile_width = 1024, uindex_t tile_height = 1024>
class TilingExecutor : public SingleContextExecutor<TransFunc, TDVS> {
  public:
    using Cell = typename TransFunc::Cell;

    /**
     * \brief The number of cells that have be added to the tile in every direction to form the
     * complete input.
     */

    static constexpr uindex_t halo_radius = TransFunc::stencil_radius * n_processing_elements;

    /**
     * \brief Create a new stencil executor.
     *
     * \param halo_value The value of cells in the grid halo.
     * \param trans_func An instance of the transition function type.
     */
    TilingExecutor(Cell halo_value, TransFunc trans_func)
        : SingleContextExecutor<TransFunc, TDVS>(halo_value, trans_func),
          input_grid(cl::sycl::buffer<Cell, 2>(cl::sycl::range<2>(0, 0))) {}

    TilingExecutor(Cell halo_value, TransFunc trans_func, TDVS tdvs)
        : SingleContextExecutor<TransFunc, TDVS>(halo_value, trans_func, tdvs),
          input_grid(cl::sycl::buffer<Cell, 2>(cl::sycl::range<2>(0, 0))) {}

    void set_input(cl::sycl::buffer<Cell, 2> input_buffer) override {
        this->input_grid = GridImpl(input_buffer);
    }

    void copy_output(cl::sycl::buffer<Cell, 2> output_buffer) override {
        input_grid.copy_to_buffer(output_buffer);
    }

    UID get_grid_range() const override { return input_grid.get_grid_range(); }

    void run(uindex_t n_generations) override {
        using in_pipe = cl::sycl::pipe<class tiling_in_pipe, Cell>;
        using out_pipe = cl::sycl::pipe<class tiling_out_pipe, Cell>;

        using ExecutionKernelImpl =
            tiling::ExecutionKernel<TransFunc, typename TDVS::KernelArgument, n_processing_elements,
                                    tile_width, tile_height, in_pipe, out_pipe>;

        this->get_tdvs().prepare_range(this->get_i_generation(), n_generations);

        cl::sycl::queue work_queue = this->new_queue();

        uindex_t target_i_generation = this->get_i_generation() + n_generations;
        uindex_t grid_width = input_grid.get_grid_range().c;
        uindex_t grid_height = input_grid.get_grid_range().r;

        while (this->get_i_generation() < target_i_generation) {
            GridImpl output_grid = input_grid.make_output_grid();

            uindex_t delta_n_generations = std::min(target_i_generation - this->get_i_generation(),
                                                    ExecutionKernelImpl::gens_per_pass);

            std::vector<cl::sycl::event> events;
            events.reserve(input_grid.get_tile_range().c * input_grid.get_tile_range().r);

            for (uindex_t tile_c = 0; tile_c < input_grid.get_tile_range().c; tile_c++) {
                for (uindex_t tile_r = 0; tile_r < input_grid.get_tile_range().r; tile_r++) {
                    input_grid.template submit_read<in_pipe>(work_queue, tile_c, tile_r);

                    cl::sycl::event computation_event =
                        work_queue.submit([&](cl::sycl::handler &cgh) {
                            auto global_state = this->get_tdvs().build_kernel_argument(
                                cgh, this->get_i_generation(), delta_n_generations);

                            cgh.single_task(ExecutionKernelImpl(
                                this->get_trans_func(), this->get_i_generation(),
                                target_i_generation, tile_c * tile_width, tile_r * tile_height,
                                grid_width, grid_height, this->get_halo_value(), global_state));
                        });
                    events.push_back(computation_event);

                    output_grid.template submit_write<out_pipe>(work_queue, tile_c, tile_r);
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

            this->inc_i_generation(delta_n_generations);
        }
    }

  private:
    using GridImpl = tiling::TiledGrid<Cell, tile_width, tile_height, halo_radius>;
    GridImpl input_grid;
};

} // namespace stencil