/*
 * Copyright © 2020-2023 Jan-Oliver Opdenhövel, Paderborn Center for Parallel
 * Computing, Paderborn University
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
#include "SingleContextExecutor.hpp"
#include "monotile/ExecutionKernel.hpp"
#include "monotile/MonotileGrid.hpp"
#include <boost/preprocessor/cat.hpp>

#include <numeric>

namespace stencil {

template <TransitionFunction TransFunc> class MonotileInputKernel;

template <TransitionFunction TransFunc> class MonotileOutputKernel;

template <TransitionFunction TransFunc, tdv::HostState TDVS, uindex_t n_processing_elements = 1,
          uindex_t tile_width = 1024, uindex_t tile_height = 1024, uindex_t word_size = 64>
/**
 * \brief An executor that follows \ref monotile.
 *
 * The feature that distincts this executor from \ref StencilExecutor is that it works with exactly
 * one tile. This means the grid range may not exceed the set tile range, but it uses less resources
 * and time per kernel execution.
 *
 * \tparam TransFunc The type of the transition function.
 * \tparam n_processing_elements The number of processing elements per kernel. Must be at least 1.
 * Defaults to 1.
 * \tparam tile_width The number of columns in a tile and maximum number of columns in a grid.
 * Defaults to 1024.
 * \tparam tile_height The number of rows in a tile and maximum number of rows in a grid. Defaults
 * to 1024.
 */
class MonotileExecutor : public SingleContextExecutor<TransFunc, TDVS> {
  public:
    using Cell = typename TransFunc::Cell;
    using GridImpl = monotile::MonotileGrid<Cell, tile_width, tile_height, word_size>;

    /**
     * \brief Create a new executor.
     *
     * \param trans_func An instance of the transition function type.
     */
    MonotileExecutor(Cell halo_value, TransFunc trans_func)
        : SingleContextExecutor<TransFunc, TDVS>(halo_value, trans_func), grid(1, 1, 0) {}

    MonotileExecutor(Cell halo_value, TransFunc trans_func, TDVS tdvs)
        : SingleContextExecutor<TransFunc, TDVS>(halo_value, trans_func, tdvs), grid(1, 1, 0) {}

    /**
     * \brief Set the internal state of the grid.
     *
     * This will copy the contents of the buffer to an internal representation. The buffer may be
     * used for other purposes later. It must not reset the generation index. The range of the input
     * buffer will be used as the new grid range.
     *
     * \throws std::range_error Thrown if the number of width or height of the buffer exceeds the
     * set width and height of the tile. \param input_buffer The source buffer of the new grid
     * state.
     */
    void set_input(cl::sycl::buffer<Cell, 2> input_buffer) override {
        grid = GridImpl(input_buffer.get_range()[0], input_buffer.get_range()[1], 0);
        grid.copy_from_buffer(input_buffer);
    }

    void copy_output(cl::sycl::buffer<Cell, 2> output_buffer) override {
        grid.copy_to_buffer(output_buffer);
    }

    UID get_grid_range() const override {
        return UID(grid.get_grid_width(), grid.get_grid_height());
    }

    void run(uindex_t n_generations) override {
        using in_pipe = cl::sycl::pipe<class monotile_in_pipe, Cell>;
        using out_pipe = cl::sycl::pipe<class monotile_out_pipe, Cell>;
        using ExecutionKernelImpl =
            monotile::ExecutionKernel<TransFunc, typename TDVS::KernelArgument,
                                      n_processing_elements, tile_width, tile_height, in_pipe,
                                      out_pipe>;

        this->get_tdvs().prepare_range(this->get_i_generation(), n_generations);

        cl::sycl::queue queue = this->new_queue();

        uindex_t target_i_generation = this->get_i_generation() + n_generations;
        uindex_t grid_width = grid.get_grid_width();
        uindex_t grid_height = grid.get_grid_height();

        GridImpl read_buffer = grid;
        GridImpl write_buffer = grid.make_similar();

        while (this->get_i_generation() < target_i_generation) {
            uindex_t delta_n_generations = std::min(target_i_generation - this->get_i_generation(),
                                                    uindex_t(ExecutionKernelImpl::gens_per_pass));

            read_buffer.template submit_read<in_pipe>(queue);

            cl::sycl::event computation_event = queue.submit([&](cl::sycl::handler &cgh) {
                auto tdv_global_state = this->get_tdvs().build_kernel_argument(
                    cgh, this->get_i_generation(), delta_n_generations);

                cgh.single_task(ExecutionKernelImpl(
                    this->get_trans_func(), this->get_i_generation(), target_i_generation,
                    grid_width, grid_height, this->get_halo_value(), tdv_global_state));
            });

            write_buffer.template submit_write<out_pipe>(queue);
            std::swap(read_buffer, write_buffer);
            this->get_runtime_sample().add_pass(computation_event);
            this->inc_i_generation(delta_n_generations);
        }

        grid = read_buffer;
    }

  private:
    GridImpl grid;
};

} // namespace stencil