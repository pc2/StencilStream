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
#include "../Concepts.hpp"
#include "../Stencil.hpp"
#include "../tdv/NoneSupplier.hpp"
#include "Grid.hpp"

namespace stencil {
namespace cpu {
template <concepts::TransitionFunction F, concepts::tdv::HostState TDVHostState = tdv::NoneSupplier>
class StencilUpdate {
  public:
    using Cell = F::Cell;
    using GridImpl = Grid<Cell>;

    struct Params {
        F transition_function;
        Cell halo_value = Cell();
        uindex_t n_generations = 1;
        TDVHostState tdv_host_state;
        sycl::queue queue = sycl::queue();
    };

    StencilUpdate(Params params) : params(params) {}

    GridImpl operator()(GridImpl &source_grid) {
        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();
        GridImpl &pass_source = source_grid;
        GridImpl &pass_target = swap_grid_b;

        for (uindex_t i_gen = 0; i_gen < params.n_generations; i_gen++) {
            for (uindex_t i_subgen = 0; i_subgen < F::n_subgenerations; i_subgen++) {
                run_gen(pass_source, pass_target, i_gen, i_subgen);
                if (i_gen == 0 && i_subgen == 0) {
                    pass_source = pass_target;
                    pass_target = swap_grid_a;
                } else {
                    std::swap(pass_source, pass_target);
                }
            }
        }

        params.queue.wait();

        return pass_source;
    }

  private:
    void run_gen(GridImpl &pass_source, GridImpl &pass_target, uindex_t i_gen, uindex_t i_subgen) {
        using TDVKernelArgument = typename TDVHostState::KernelArgument;
        using TDVLocalState = typename TDVKernelArgument::LocalState;
        using TDVValue = typename TDVLocalState::Value;

        params.queue.submit([&](sycl::handler &cgh) {
            sycl::accessor source_ac(pass_source.get_buffer(), cgh, sycl::read_only);
            sycl::accessor target_ac(pass_target.get_buffer(), cgh, sycl::write_only);
            index_t grid_width = source_ac.get_range()[0];
            index_t grid_height = source_ac.get_range()[1];
            index_t stencil_radius = index_t(F::stencil_radius);
            Cell halo_value = params.halo_value;
            F transition_function = params.transition_function;
            TDVKernelArgument tdv_kernel_argument =
                params.tdv_host_state.build_kernel_argument(cgh, i_gen, 1);
            TDVLocalState tdv_local_state = tdv_kernel_argument.build_local_state();
            TDVValue tdv_value = tdv_local_state.get_value(0);

            auto kernel = [=](sycl::id<2> id) {
                using StencilImpl = Stencil<Cell, F::stencil_radius, TDVValue>;
                StencilImpl stencil(ID(id[0], id[1]), UID(grid_width, grid_height), i_gen, i_subgen,
                                    i_subgen, tdv_value);

                for (index_t rel_c = -stencil_radius; rel_c <= stencil_radius; rel_c++) {
                    for (index_t rel_r = -stencil_radius; rel_r <= stencil_radius; rel_r++) {
                        index_t c = rel_c + id[0];
                        index_t r = rel_r + id[1];
                        bool within_grid = c >= 0 && r >= 0 && c < grid_width && r < grid_height;
                        stencil[ID(rel_c, rel_r)] = (within_grid) ? source_ac[c][r] : halo_value;
                    }
                }

                target_ac[id] = transition_function(stencil);
            };

            cgh.parallel_for(source_ac.get_range(), kernel);
        });
    }

    Params params;
};
} // namespace cpu
} // namespace stencil