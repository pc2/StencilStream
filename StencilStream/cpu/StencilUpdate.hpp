/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
 * University
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

/**
 * \brief A grid updater that applies an iterative stencil code to a grid.
 *
 * This updater applies an iterative stencil code, defined by the template parameter `F`, to the
 * grid; As often as requested.
 *
 * \tparam F The transition function to apply to input grids.
 */
template <concepts::TransitionFunction F> class StencilUpdate {
  private:
    using Cell = F::Cell;

  public:
    /// \brief Shorthand for the used and supported grid type.
    using GridImpl = Grid<Cell>;

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
        uindex_t iteration_offset = 0;

        /**
         * \brief The number of iterations to compute.
         */
        uindex_t n_iterations = 1;

        /**
         * \brief The device to use for computations.
         */
        sycl::device device = sycl::device();

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
    };

    /**
     * \brief Create a new stencil updater object.
     */
    StencilUpdate(Params params) : params(params), n_processed_cells(0), walltime(0.0) {}

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
        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();
        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        sycl::queue queue(params.device);
        auto walltime_start = std::chrono::high_resolution_clock::now();

        for (uindex_t i_iter = 0; i_iter < params.n_iterations; i_iter++) {
            for (uindex_t i_subiter = 0; i_subiter < F::n_subiterations; i_subiter++) {
                run_iter(queue, pass_source, pass_target, params.iteration_offset + i_iter,
                         i_subiter);
                if (i_iter == 0 && i_subiter == 0) {
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
            params.n_iterations * source_grid.get_grid_width() * source_grid.get_grid_height();

        return *pass_source;
    }

    /**
     * \brief Return a reference to the parameters.
     *
     * Modifications to the parameters struct will be used in the next call to \ref operator()().
     */
    Params &get_params() { return params; }

    /**
     * \brief Return the accumulated total number of cells processed by this updater.
     *
     * For each call of to \ref operator()(), this is the width times the height of the grid, times
     * the number of computed iterations. This will also be accumulated across multiple calls to
     * \ref operator()().
     */
    uindex_t get_n_processed_cells() const { return n_processed_cells; }

    /**
     * \brief Return the accumulated runtime of the updater, measured from the host side.
     *
     * For each call to \ref operator()(), the time it took to submit all kernels and, if \ref
     * Params::blocking is true, to finish the computation is recorded and accumulated.
     */
    double get_walltime() const { return walltime; }

  private:
    /**
     * \brief Update the source grid by one iteration.
     *
     * This method will read the current state of the grid from the pass source and write the update
     * the pass target.
     *
     * \param queue The queue to submit the kernel to.
     *
     * \param pass_source A pointer to a grid. The old state of the grid will be read from here.
     *
     * \param pass_target A pointer to a grid. The new state will be written to this grid.
     *
     * \param i_iter The index of the iteration to compute.
     *
     * \param i_subiter The index of the sub-iteration to compute.
     */
    void run_iter(sycl::queue queue, GridImpl *pass_source, GridImpl *pass_target, uindex_t i_iter,
                  uindex_t i_subiter) {
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
            TDV tdv = transition_function.get_time_dependent_value(i_iter);

            auto kernel = [=](sycl::id<2> id) {
                StencilImpl stencil(ID(id[0], id[1]), UID(grid_width, grid_height), i_iter,
                                    i_subiter, tdv);

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