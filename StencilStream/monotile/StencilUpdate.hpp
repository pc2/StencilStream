/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "../GenericID.hpp"
#include "../Helpers.hpp"
#include "../Index.hpp"
#include "../tdv/SinglePassStrategies.hpp"
#include "Grid.hpp"
#include <chrono>
#include <type_traits>

namespace stencil {
namespace monotile {

/**
 * \brief A kernel that executes a stencil transition function using the monotile approach.
 *
 * It receives the contents of a tile and it's halo from the `in_pipe`, applies the transition
 * function when applicable and writes the result to the `out_pipe`.
 *
 * With the monotile approach, the whole grid fits in one tile. This eliminates the need to
 * calculate the cells of the tile halo, reducing the cache size and number of loop iterations. More
 * is described in \ref monotile.
 *
 * \tparam TransFunc The type of transition function to use.
 * \tparam n_processing_elements The number of processing elements to use. Similar to an unroll
 * factor for a loop.
 * \tparam output_tile_width The number of columns in a grid tile.
 * \tparam output_tile_height The number of rows in a grid tile.
 * \tparam in_pipe The pipe to read from.
 * \tparam out_pipe The pipe to write to.
 */
template <concepts::TransitionFunction TransFunc,
          tdv::single_pass::KernelArgument<TransFunc> TDVKernelArgument,
          uindex_t n_processing_elements, uindex_t max_grid_width, uindex_t max_grid_height,
          typename in_pipe, typename out_pipe>
    requires(n_processing_elements % TransFunc::n_subiterations == 0)
class StencilUpdateKernel {
  public:
    using Cell = typename TransFunc::Cell;
    using TDV = typename TransFunc::TimeDependentValue;

    using TDVLocalState = typename TDVKernelArgument::LocalState;
    using StencilImpl = Stencil<Cell, TransFunc::stencil_radius, TDV>;

    /**
     * \brief The width and height of the stencil buffer.
     */
    static constexpr uindex_t stencil_diameter = StencilImpl::diameter;

    static constexpr uindex_t iters_per_pass = n_processing_elements / TransFunc::n_subiterations;

    static constexpr uindex_t calc_pipeline_latency(uindex_t grid_height) {
        return n_processing_elements * TransFunc::stencil_radius * (grid_height + 1);
    }

    static constexpr uindex_t calc_n_iterations(uindex_t grid_width, uindex_t grid_height) {
        return grid_width * grid_height + calc_pipeline_latency(grid_height);
    }

    using index_stencil_t = typename StencilImpl::index_stencil_t;
    using uindex_stencil_t = typename StencilImpl::uindex_stencil_t;
    using StencilID = typename StencilImpl::StencilID;
    using StencilUID = typename StencilImpl::StencilUID;

    static constexpr unsigned long bits_1d =
        std::bit_width(std::max(max_grid_width, max_grid_height));
    using index_1d_t = ac_int<bits_1d + 1, true>;
    using uindex_1d_t = ac_int<bits_1d, false>;

    static constexpr unsigned long bits_2d = 2 * bits_1d;
    using index_2d_t = ac_int<bits_2d + 1, true>;
    using uindex_2d_t = ac_int<bits_2d, false>;

    static constexpr unsigned long bits_pes =
        std::max<int>(2, std::bit_width(n_processing_elements));
    using index_pes_t = ac_int<bits_pes + 1, true>;
    using uindex_pes_t = ac_int<bits_pes, false>;

    static constexpr unsigned long bits_n_iterations =
        std::bit_width(calc_n_iterations(max_grid_width, max_grid_height));
    using index_n_iterations_t = ac_int<bits_n_iterations + 1, true>;
    using uindex_n_iterations_t = ac_int<bits_n_iterations, false>;

    /**
     * \brief Create and configure the execution kernel.
     *
     * \param trans_func The instance of the transition function to use.
     * \param i_iteration The iteration index of the input cells.
     * \param n_iterations The number of iterations to compute. If this number is bigger than
     * `n_processing_elements`, only `n_processing_elements` iterations will be computed.
     * \param grid_width The number of cell columns in the grid.
     * \param grid_height The number of cell rows in the grid.
     * \param halo_value The value of cells outside the grid.
     */
    StencilUpdateKernel(TransFunc trans_func, uindex_t i_iteration, uindex_t target_i_iteration,
                        uindex_t grid_width, uindex_t grid_height, Cell halo_value,
                        TDVKernelArgument tdv_kernel_argument)
        : trans_func(trans_func), i_iteration(i_iteration), target_i_iteration(target_i_iteration),
          grid_width(grid_width), grid_height(grid_height), halo_value(halo_value),
          tdv_kernel_argument(tdv_kernel_argument) {
        assert(grid_height <= max_grid_height);
    }

    /**
     * \brief Execute the kernel.
     */
    void operator()() const {
        [[intel::fpga_register]] index_1d_t c[n_processing_elements];
        [[intel::fpga_register]] index_1d_t r[n_processing_elements];
        TDVLocalState tdv_local_state(tdv_kernel_argument);

        // Initializing (output) column and row counters.
        index_1d_t prev_c = 0;
        index_1d_t prev_r = 0;
#pragma unroll
        for (uindex_pes_t i = 0; i < uindex_pes_t(n_processing_elements); i++) {
            c[i] = prev_c - TransFunc::stencil_radius;
            r[i] = prev_r - TransFunc::stencil_radius;
            if (r[i] < index_pes_t(0)) {
                r[i] += grid_height;
                c[i] -= 1;
            }
            prev_c = c[i];
            prev_r = r[i];
        }

        /*
         * The intel::numbanks attribute requires a power of two as it's argument and if the
         * number of processing elements isn't a power of two, it would produce an error. Therefore,
         * we calculate the next power of two and use it to allocate the cache. The compiler is
         * smart enough to see that these additional banks in the cache aren't used and therefore
         * optimizes them away.
         */
        [[intel::fpga_memory,
          intel::numbanks(2 * std::bit_ceil(n_processing_elements))]] Padded<Cell>
            cache[2][max_grid_height][std::bit_ceil(n_processing_elements)][stencil_diameter - 1];
        [[intel::fpga_register]] Cell stencil_buffer[n_processing_elements][stencil_diameter]
                                                    [stencil_diameter];

        uindex_n_iterations_t n_iterations = calc_n_iterations(grid_width, grid_height);
        for (uindex_n_iterations_t i = 0; i < n_iterations; i++) {
            Cell carry;
            if (i < uindex_n_iterations_t(grid_width * grid_height)) {
                carry = in_pipe::read();
            } else {
                carry = halo_value;
            }

#pragma unroll
            for (uindex_pes_t i_processing_element = 0;
                 i_processing_element < uindex_pes_t(n_processing_elements);
                 i_processing_element++) {
#pragma unroll
                for (uindex_stencil_t r = 0; r < uindex_stencil_t(stencil_diameter - 1); r++) {
#pragma unroll
                    for (uindex_stencil_t c = 0; c < uindex_stencil_t(stencil_diameter); c++) {
                        stencil_buffer[i_processing_element][c][r] =
                            stencil_buffer[i_processing_element][c][r + 1];
                    }
                }

                // Update the stencil buffer and cache with previous cache contents and the new
                // input cell.
#pragma unroll
                for (uindex_stencil_t cache_c = 0; cache_c < uindex_stencil_t(stencil_diameter);
                     cache_c++) {
                    Cell new_value;
                    if (cache_c == uindex_stencil_t(stencil_diameter - 1)) {
                        new_value = carry;
                    } else {
                        new_value = cache[c[i_processing_element][0]][r[i_processing_element]]
                                         [i_processing_element][cache_c]
                                             .value;
                    }

                    stencil_buffer[i_processing_element][cache_c][stencil_diameter - 1] = new_value;
                    if (cache_c > 0) {
                        cache[(~c[i_processing_element])[0]][r[i_processing_element]]
                             [i_processing_element][cache_c - 1]
                                 .value = new_value;
                    }
                }

                uindex_t pe_iteration =
                    (i_iteration + i_processing_element / TransFunc::n_subiterations).to_uint();
                uindex_t pe_subiteration =
                    (i_processing_element % TransFunc::n_subiterations).to_uint();

                if (pe_iteration < target_i_iteration) {
                    TDV tdv = tdv_local_state.get_time_dependent_value(
                        (i_processing_element / TransFunc::n_subiterations).to_uint());
                    StencilImpl stencil(ID(c[i_processing_element], r[i_processing_element]),
                                        UID(grid_width, grid_height), pe_iteration, pe_subiteration,
                                        tdv);

                    bool h_halo_mask[stencil_diameter];
                    bool v_halo_mask[stencil_diameter];
#pragma unroll
                    for (uindex_stencil_t mask_i = 0; mask_i < uindex_stencil_t(stencil_diameter);
                         mask_i++) {
                        // These computation assume that the central cell is in the grid. If it's
                        // not, the resulting value of this processing element will be discarded
                        // anyways, so this is safe.
                        if (mask_i < uindex_stencil_t(TransFunc::stencil_radius)) {
                            h_halo_mask[mask_i] = c[i_processing_element] >=
                                                  index_1d_t(TransFunc::stencil_radius - mask_i);
                            v_halo_mask[mask_i] = r[i_processing_element] >=
                                                  index_1d_t(TransFunc::stencil_radius - mask_i);
                        } else if (mask_i == uindex_stencil_t(TransFunc::stencil_radius)) {
                            h_halo_mask[mask_i] = true;
                            v_halo_mask[mask_i] = true;
                        } else {
                            h_halo_mask[mask_i] =
                                c[i_processing_element] <
                                grid_width + index_1d_t(TransFunc::stencil_radius - mask_i);
                            v_halo_mask[mask_i] =
                                r[i_processing_element] <
                                grid_height + index_1d_t(TransFunc::stencil_radius - mask_i);
                        }
                    }

#pragma unroll
                    for (uindex_stencil_t cell_c = 0; cell_c < uindex_stencil_t(stencil_diameter);
                         cell_c++) {
#pragma unroll
                        for (uindex_stencil_t cell_r = 0;
                             cell_r < uindex_stencil_t(stencil_diameter); cell_r++) {
                            if (h_halo_mask[cell_c] && v_halo_mask[cell_r]) {
                                stencil[StencilUID(cell_c, cell_r)] =
                                    stencil_buffer[i_processing_element][cell_c][cell_r];
                            } else {
                                stencil[StencilUID(cell_c, cell_r)] = halo_value;
                            }
                        }
                    }

                    carry = trans_func(stencil);
                } else {
                    carry = stencil_buffer[i_processing_element][TransFunc::stencil_radius]
                                          [TransFunc::stencil_radius];
                }

                r[i_processing_element] += 1;
                if (r[i_processing_element] == index_1d_t(grid_height)) {
                    r[i_processing_element] = 0;
                    c[i_processing_element] += 1;
                }
            }

            if (i >= uindex_n_iterations_t(calc_pipeline_latency(grid_height))) {
                out_pipe::write(carry);
            }
        }
    }

  private:
    TransFunc trans_func;
    uindex_t i_iteration;
    uindex_t target_i_iteration;
    uindex_t grid_width;
    uindex_t grid_height;
    Cell halo_value;
    TDVKernelArgument tdv_kernel_argument;
};

template <concepts::TransitionFunction F, uindex_t n_processing_elements = 1,
          uindex_t max_grid_width = 1024, uindex_t max_grid_height = 1024,
          tdv::single_pass::Strategy<F, n_processing_elements> TDVStrategy =
              tdv::single_pass::InlineStrategy,
          uindex_t word_size = 64>
class StencilUpdate {
  public:
    using Cell = F::Cell;
    using TDV = typename F::TimeDependentValue;
    using GridImpl = Grid<Cell, word_size>;

    struct Params {
        F transition_function;
        Cell halo_value = Cell();
        uindex_t iteration_offset = 0;
        uindex_t n_iterations = 1;
        sycl::device device = sycl::device();
        bool blocking = false;
        bool profiling = false;
    };

    StencilUpdate(Params params)
        : params(params), n_processed_cells(0), work_events(), walltime(0.0) {}

    Params &get_params() { return params; }

    GridImpl operator()(GridImpl &source_grid) {
        if (source_grid.get_grid_height() > max_grid_height) {
            throw std::range_error("The grid is too tall for the stencil update kernel.");
        }
        if (source_grid.get_grid_width() > max_grid_width) {
            throw std::range_error("The grid is too wide for the stencil update kernel.");
        }
        using in_pipe = sycl::pipe<class monotile_in_pipe, Cell>;
        using out_pipe = sycl::pipe<class monotile_out_pipe, Cell>;

        constexpr uindex_t iters_per_pass = n_processing_elements / F::n_subiterations;

        using TDVGlobalState = TDVStrategy::template GlobalState<F, iters_per_pass>;
        using TDVKernelArgument = typename TDVGlobalState::KernelArgument;
        using ExecutionKernelImpl =
            StencilUpdateKernel<F, TDVKernelArgument, n_processing_elements, max_grid_width,
                                max_grid_height, in_pipe, out_pipe>;

        sycl::queue input_kernel_queue =
            sycl::queue(params.device, {sycl::property::queue::in_order{}});
        sycl::queue output_kernel_queue =
            sycl::queue(params.device, {sycl::property::queue::in_order{}});

        sycl::queue update_kernel_queue =
            sycl::queue(params.device, {cl::sycl::property::queue::enable_profiling{},
                                        sycl::property::queue::in_order{}});

        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();

        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        F trans_func = params.transition_function;
        TDVGlobalState tdv_global_state(trans_func, params.iteration_offset, params.n_iterations);

        auto walltime_start = std::chrono::high_resolution_clock::now();

        uindex_t target_n_iterations = params.iteration_offset + params.n_iterations;
        for (uindex_t i = params.iteration_offset; i < target_n_iterations; i += iters_per_pass) {
            pass_source->template submit_read<in_pipe>(input_kernel_queue);
            uindex_t iters_in_this_pass = std::min(iters_per_pass, target_n_iterations - i);

            sycl::event work_event = update_kernel_queue.submit([&](sycl::handler &cgh) {
                TDVKernelArgument tdv_kernel_argument(tdv_global_state, cgh, i, iters_in_this_pass);
                ExecutionKernelImpl exec_kernel(
                    trans_func, i, target_n_iterations, source_grid.get_grid_width(),
                    source_grid.get_grid_height(), params.halo_value, tdv_kernel_argument);
                cgh.single_task<ExecutionKernelImpl>(exec_kernel);
            });
            if (params.profiling) {
                work_events.push_back(work_event);
            }

            pass_target->template submit_write<out_pipe>(output_kernel_queue);

            if (i == params.iteration_offset) {
                pass_source = &swap_grid_b;
                pass_target = &swap_grid_a;
            } else {
                std::swap(pass_source, pass_target);
            }
        }

        if (params.blocking) {
            output_kernel_queue.wait();
        }

        auto walltime_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> walltime = walltime_end - walltime_start;
        this->walltime += walltime.count();

        n_processed_cells +=
            params.n_iterations * source_grid.get_grid_width() * source_grid.get_grid_height();

        return *pass_source;
    }

    uindex_t get_n_processed_cells() const { return n_processed_cells; }

    double get_kernel_runtime() const {
        double kernel_runtime = 0.0;
        for (sycl::event work_event : work_events) {
            const double timesteps_per_second = 1000000000.0;
            double start =
                double(work_event
                           .get_profiling_info<cl::sycl::info::event_profiling::command_start>()) /
                timesteps_per_second;
            double end =
                double(
                    work_event.get_profiling_info<cl::sycl::info::event_profiling::command_end>()) /
                timesteps_per_second;
            kernel_runtime += end - start;
        }
        return kernel_runtime;
    }

    double get_walltime() const { return walltime; }

  private:
    Params params;
    uindex_t n_processed_cells;
    double walltime;
    std::vector<sycl::event> work_events;
};

} // namespace monotile
} // namespace stencil