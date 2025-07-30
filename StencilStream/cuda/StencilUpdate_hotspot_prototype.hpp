#pragma once
#include "../Stencil_prototype.hpp"
#include "Grid_hotspot_prototype.hpp"
#include <chrono>
#include <sycl/sycl.hpp>
#include <vector>

#include <bit>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/id.hpp>
#include <sycl/range.hpp>
#include <variant>

// StencilUpdater with GPU data conversion

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

        bool blocking = true;
        bool profiling = true;
    };

    StencilUpdate(Params params)
        : params(params), n_processed_cells(0), walltime(0.0), work_events() {}

    GridImpl operator()(GridImpl &source_grid) {
        sycl::queue update_kernel_queue =
            sycl::queue(params.device, {sycl::property::queue::enable_profiling()});

        auto data_preperation_start_before = std::chrono::high_resolution_clock::now();

        // Scatter the Grid into Arrays

        update_kernel_queue.submit([&](sycl::handler &cgh) {
            sycl::accessor ac_source_grid(source_grid.get_buffer(), cgh);

            sycl::accessor temp_out_ac_source_grid(source_grid.get_temp_out_buffer(), cgh,
                                                   sycl::write_only);
            sycl::accessor power_out_ac_source_grid(source_grid.get_power_out_buffer(), cgh,
                                                    sycl::write_only);

            cgh.parallel_for(ac_source_grid.get_range(), [=](sycl::id<2> id) {
                Cell cell = ac_source_grid[id[0]][id[1]];
                size_t cell_id = id[0] * ac_source_grid.get_range()[1] + id[1];

                temp_out_ac_source_grid[cell_id] = cell[0];
                power_out_ac_source_grid[cell_id] = cell[1];
            });
        });

        update_kernel_queue.wait();

        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();
        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        auto data_preperation_end_before = std::chrono::high_resolution_clock::now();

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

        // auto data_preperation_start_after = std::chrono::high_resolution_clock::now();

        // Gather the Arrays into Grid

        update_kernel_queue.submit([&](sycl::handler &cgh) {
            sycl::accessor ac_pass_grid(pass_source->get_buffer(), cgh, sycl::write_only);

            sycl::accessor temp_out_ac_pass_grid(pass_source->get_temp_out_buffer(), cgh,
                                                 sycl::read_only);
            sycl::accessor power_out_ac_pass_grid(pass_source->get_power_out_buffer(), cgh,
                                                  sycl::read_only);

            auto kernel = [=](sycl::id<2> id) {
                Cell cell;
                size_t cell_id = id[0] * ac_pass_grid.get_range()[1] + id[1];

                cell[0] = temp_out_ac_pass_grid[cell_id];
                cell[1] = power_out_ac_pass_grid[cell_id];

                ac_pass_grid[id[0]][id[1]] = cell;
            };

            cgh.parallel_for(ac_pass_grid.get_range(), kernel);
        });
        update_kernel_queue.wait();

        auto data_preperation_end_after = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> data_preperation_time_before =
            data_preperation_end_before - data_preperation_start_before;

        std::chrono::duration<double> data_preperation_time_after =
            data_preperation_end_after - data_preperation_start_after;

        this->data_preperation_time_before += data_preperation_time_before.count();
        this->data_preperation_time_after += data_preperation_time_after.count();

        n_processed_cells +=
            params.n_iterations * source_grid.get_grid_height() * source_grid.get_grid_width();

        return *pass_source;
    }

    std::size_t get_n_processed_cells() const { return n_processed_cells; }

    double get_walltime() const { return walltime; }
    double get_data_preperation_time_before() const { return data_preperation_time_before; }
    double get_data_preperation_time_after() const { return data_preperation_time_after; }

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

            sycl::accessor temp_out_ac_pass_source(pass_source->get_temp_out_buffer(), cgh,
                                                   sycl::read_only);
            sycl::accessor power_out_ac_pass_source(pass_source->get_power_out_buffer(), cgh,
                                                    sycl::read_only);

            sycl::accessor temp_out_ac_pass_target(pass_target->get_temp_out_buffer(), cgh,
                                                   sycl::write_only);
            sycl::accessor power_out_ac_pass_target(pass_target->get_power_out_buffer(), cgh,
                                                    sycl::write_only);

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
                            size_t cell_id = (id[0] + rel_r - F::stencil_radius) * grid_width +
                                             (id[1] + rel_c - F::stencil_radius);

                            cell[0] = T_out_ac_pass_source[cell_id];
                            cell[1] = Pt_out_ac_pass_source[cell_id];

                        } else {
                            cell = halo_value;
                        }
                        stencil[sycl::id<2>(rel_r, rel_c)] = cell;
                    }
                }
                size_t cell_id = id[0] * grid_width + id[1];
                Cell new_cell = transition_function(stencil);
                temp_out_ac_pass_target[cell_id] = new_cell[0];
                power_out_ac_pass_target[cell_id] = new_cell[1];
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
    double data_preperation_time_before;
    double data_preperation_time_after;
};
