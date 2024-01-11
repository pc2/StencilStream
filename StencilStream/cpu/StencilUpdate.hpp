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
#include "../Concepts.hpp"
#include "../Stencil.hpp"
#include "Grid.hpp"
#include <chrono>

namespace stencil {
namespace cpu {
template <concepts::TransitionFunction F> class StencilUpdate {
  public:
    using Cell = F::Cell;
    using GridImpl = Grid<Cell>;

    struct Params {
        F transition_function;
        Cell halo_value = Cell();
        uindex_t generation_offset = 0;
        uindex_t n_generations = 1;
        sycl::device device = sycl::device();
        bool blocking = false;
    };

    StencilUpdate(Params params) : params(params), n_processed_cells(0), walltime(0.0) {}

    GridImpl operator()(GridImpl &source_grid) {
        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();
        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        sycl::queue queue(params.device);
        auto walltime_start = std::chrono::high_resolution_clock::now();

        for (uindex_t i_gen = 0; i_gen < params.n_generations; i_gen++) {
            for (uindex_t i_subgen = 0; i_subgen < F::n_subgenerations; i_subgen++) {
                run_gen(queue, pass_source, pass_target, params.generation_offset + i_gen,
                        i_subgen);
                if (i_gen == 0 && i_subgen == 0) {
                    pass_source = &swap_grid_b;
                    pass_target = &swap_grid_a;
                } else {
                    std::swap(pass_source, pass_target);
                }
            }
        }

        if (params.blocking) {
            queue.wait();
        }

        auto walltime_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> walltime = walltime_end - walltime_start;
        this->walltime += walltime.count();
        n_processed_cells +=
            params.n_generations * source_grid.get_grid_width() * source_grid.get_grid_height();

        return *pass_source;
    }

    Params &get_params() { return params; }

    uindex_t get_n_processed_cells() const { return n_processed_cells; }

    double get_walltime() const { return walltime; }

  private:
    void run_gen(sycl::queue queue, GridImpl *pass_source, GridImpl *pass_target, uindex_t i_gen,
                 uindex_t i_subgen) {
        using TDV = typename F::TimeDependentValue;
        using StencilImpl = Stencil<Cell, F::stencil_radius, TDV>;

        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor source_ac(pass_source->get_buffer(), cgh, sycl::read_only);
            sycl::accessor target_ac(pass_target->get_buffer(), cgh, sycl::write_only);
            index_t grid_width = source_ac.get_range()[0];
            index_t grid_height = source_ac.get_range()[1];
            index_t stencil_radius = index_t(F::stencil_radius);
            Cell halo_value = params.halo_value;
            F transition_function = params.transition_function;
            TDV tdv = transition_function.get_time_dependent_value(i_gen);

            auto kernel = [=](sycl::id<2> id) {
                StencilImpl stencil(ID(id[0], id[1]), UID(grid_width, grid_height), i_gen, i_subgen,
                                    i_subgen, tdv);

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
    uindex_t n_processed_cells;
    double walltime;
};
} // namespace cpu
} // namespace stencil