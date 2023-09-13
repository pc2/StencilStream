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
#include "Grid.hpp"

namespace stencil {
namespace cpu {
template <concepts::TransitionFunction F> class StencilUpdate {
  public:
    using Cell = F::Cell;
    using GridImpl = Grid<Cell>;

    struct Params {
        F transition_function;
        Cell halo_value = Cell();
        uindex_t n_generations = 1;
        cl::sycl::queue queue = cl::sycl::queue();
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

        return pass_source;
    }

  private:
    void run_gen(GridImpl &pass_source, GridImpl &pass_target, uindex_t i_gen, uindex_t i_subgen) {
        params.queue.submit([&](cl::sycl::handler &cgh) {
            auto source_ac =
                pass_source.get_buffer().template get_access<cl::sycl::access::mode::read>(cgh);
            auto target_ac =
                pass_target.get_buffer().template get_access<cl::sycl::access::mode::discard_write>(
                    cgh);
            index_t grid_width = source_ac.get_range()[0];
            index_t grid_height = source_ac.get_range()[1];
            index_t stencil_radius = index_t(F::stencil_radius);
            Cell halo_value = params.halo_value;
            F transition_function = params.transition_function;

            auto kernel = [=](cl::sycl::id<2> id) {
                using StencilImpl = Stencil<Cell, F::stencil_radius>;
                StencilImpl stencil(ID(id[0], id[1]), UID(grid_width, grid_height), i_gen, i_subgen,
                                    i_subgen, std::monostate());

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