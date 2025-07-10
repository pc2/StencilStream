#pragma once
#include "../Stencil_prototype.hpp"
#include "Grid_baseline.hpp"
#include <chrono>
#include <sycl/sycl.hpp>
#include <vector>

#include <bit>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/id.hpp>
#include <sycl/range.hpp>
#include <variant>

template <typename F> class StencilUpdate {
  private:
    using Cell = typename F::Cell;
    std::vector<sycl::event> work_events;

  public:
    using GridImpl = Grid<Cell>;

    struct Params {

        F transition_function;

        Cell halo_value = Cell();

        std::size_t iteration_offset = 0;

        std::size_t n_iterations = 1;

        sycl::device device = sycl::device();

        bool blocking = false;
        bool profiling = true;
    };

    StencilUpdate(Params params)
        : params(params), n_processed_cells(0), walltime(0.0), work_events() {}

    GridImpl operator()(GridImpl &source_grid) {
        sycl::queue update_kernel_queue =
            sycl::queue(params.device, {sycl::property::queue::enable_profiling()});

        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();
        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        auto walltime_start = std::chrono::high_resolution_clock::now();

        for (std::size_t i_iter = 0; i_iter < params.n_iterations; i_iter++) {
            for (std::size_t i_subiter = 0; i_subiter < F::n_subiterations; i_subiter++) {
                run_iter(update_kernel_queue, pass_source, pass_target,
                         params.iteration_offset + i_iter, i_subiter);
                if (i_iter == 0 && i_subiter == 0) {
                    pass_source = &swap_grid_b;
                    pass_target = &swap_grid_a;
                } else {
                    std::swap(pass_source, pass_target);
                }
            }
        }

        if (params.blocking) {
            update_kernel_queue.wait();
        }

        auto walltime_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> walltime = walltime_end - walltime_start;
        this->walltime += walltime.count();
        n_processed_cells +=
            params.n_iterations * source_grid.get_grid_height() * source_grid.get_grid_width();

        return *pass_source;
    }

    std::size_t get_n_processed_cells() const { return n_processed_cells; }

    double get_walltime() const { return walltime; }

    void clear_work_events() { work_events.clear(); }

    double get_kernel_runtime() const {
        double kernel_runtime = 0.0;
        for (sycl::event work_event : work_events) {
            const double timesteps_per_second = 1000000000.0;
            double start =
                double(
                    work_event.get_profiling_info<sycl::info::event_profiling::command_start>()) /
                timesteps_per_second;
            double end =
                double(work_event.get_profiling_info<sycl::info::event_profiling::command_end>()) /
                timesteps_per_second;
            kernel_runtime += end - start;
        }
        return kernel_runtime;
    }

  private:
    void run_iter(sycl::queue queue, GridImpl *pass_source, GridImpl *pass_target,
                  std::size_t i_iter, std::size_t i_subiter) {

        using StencilImpl = Stencil<Cell, F::stencil_radius>;

        sycl::event work_event = queue.submit([&](sycl::handler &cgh) {
            sycl::accessor source_ac(pass_source->get_buffer(), cgh, sycl::read_only);
            sycl::accessor target_ac(pass_target->get_buffer(), cgh, sycl::write_only);
            std::size_t grid_height = source_ac.get_range()[0];
            std::size_t grid_width = source_ac.get_range()[1];
            Cell halo_value = params.halo_value;
            F transition_function = params.transition_function;

            auto kernel = [=](sycl::id<2> id) {
                StencilImpl stencil(id, source_ac.get_range(), i_iter, i_subiter);

                for (std::size_t rel_r = 0; rel_r < 2 * F::stencil_radius + 1; rel_r++) {
                    for (std::size_t rel_c = 0; rel_c < 2 * F::stencil_radius + 1; rel_c++) {
                        Cell cell;
                        if (id[0] + rel_r >= F::stencil_radius &&
                            id[1] + rel_c >= F::stencil_radius &&
                            id[0] + rel_r < grid_height + F::stencil_radius &&
                            id[1] + rel_c < grid_width + F::stencil_radius) {
                            cell = source_ac[id[0] + rel_r - F::stencil_radius]
                                            [id[1] + rel_c - F::stencil_radius];
                        } else {
                            cell = halo_value;
                        }
                        stencil[sycl::id<2>(rel_r, rel_c)] = cell;
                    }
                }

                target_ac[id] = transition_function(stencil);
            };

            cgh.parallel_for(source_ac.get_range(), kernel);
        });
        if (params.profiling) {
            work_events.push_back(work_event);
        }
    }

    Params params;
    std::size_t n_processed_cells;
    double walltime;
};
