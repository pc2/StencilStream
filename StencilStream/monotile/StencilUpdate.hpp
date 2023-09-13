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
#include "../tdv/NoneSupplier.hpp"
#include "ExecutionKernel.hpp"
#include "Grid.hpp"

namespace stencil {
namespace monotile {

template <concepts::TransitionFunction F, uindex_t n_processing_elements = 1,
          uindex_t tile_width = 1024, uindex_t tile_height = 1024, uindex_t word_size = 64>
class StencilUpdate {
  public:
    using Cell = F::Cell;
    using GridImpl = Grid<Cell, tile_width, tile_height, word_size>;

    struct Params {
        F transition_function;
        Cell halo_value = Cell();
        uindex_t n_generations = 1;
        cl::sycl::queue queue = cl::sycl::queue();
    };

    StencilUpdate(Params params) : params(params) {}

    GridImpl operator()(GridImpl &source_grid) {
        using in_pipe = cl::sycl::pipe<class monotile_in_pipe, Cell>;
        using out_pipe = cl::sycl::pipe<class monotile_out_pipe, Cell>;
        using ExecutionKernelImpl = ExecutionKernel<F, tdv::NoneSupplier, n_processing_elements,
                                                    tile_width, tile_height, in_pipe, out_pipe>;

        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();

        uindex_t gens_per_pass = ExecutionKernelImpl::gens_per_pass;
        GridImpl &pass_source = source_grid;
        GridImpl &pass_target = swap_grid_b;

        for (uindex_t i_gen = 0; i_gen < params.n_generations; i_gen += gens_per_pass) {
            pass_source.template submit_read<in_pipe>(params.queue);
            params.queue.submit([&](cl::sycl::handler &cgh) {
                ExecutionKernelImpl exec_kernel(params.transition_function, i_gen,
                                                params.n_generations, source_grid.get_grid_width(),
                                                source_grid.get_grid_height(), params.halo_value,
                                                tdv::NoneSupplier());
                cgh.single_task<ExecutionKernelImpl>(exec_kernel);
            });
            pass_target.template submit_write<out_pipe>(params.queue);

            if (i_gen == 0) {
                pass_source = swap_grid_b;
                pass_target = swap_grid_a;
            } else {
                std::swap(pass_source, pass_target);
            }
        }

        params.queue.wait();

        return pass_source;
    }

  private:
    Params params;
};

} // namespace monotile
} // namespace stencil