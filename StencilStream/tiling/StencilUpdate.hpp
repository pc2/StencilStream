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
#include "../Padded.hpp"
#include "Grid.hpp"

#include <chrono>
#include <optional>

namespace stencil {
namespace tiling {

/**
 * \brief A kernel that executes a stencil transition function on a tile.
 *
 * It receives the contents of a tile and it's halo from the `in_pipe`, applies the transition
 * function when applicable and writes the result to the `out_pipe`.
 *
 * \tparam TransFunc The type of transition function to use.
 * \tparam n_processing_elements The number of processing elements to use. Similar to an unroll
 * factor for a loop.
 * \tparam output_tile_width The number of columns in a grid tile.
 * \tparam output_tile_height The number of rows in a grid tile.
 * \tparam in_pipe The pipe to read from.
 * \tparam out_pipe The pipe to write to.
 */
template <concepts::TransitionFunction TransFunc, uindex_t n_processing_elements,
          uindex_t output_tile_width, uindex_t output_tile_height, typename in_pipe,
          typename out_pipe>
    requires(n_processing_elements % TransFunc::n_subgenerations == 0)
class StencilUpdateKernel {
  public:
    using Cell = typename TransFunc::Cell;
    using TDV = typename TransFunc::TimeDependentValue;
    using StencilImpl = Stencil<Cell, TransFunc::stencil_radius, TDV>;

    /**
     * \brief The width and height of the stencil buffer.
     */
    static constexpr uindex_t stencil_diameter = StencilImpl::diameter;

    static constexpr uindex_t gens_per_pass = n_processing_elements / TransFunc::n_subgenerations;

    static constexpr uindex_t halo_radius = TransFunc::stencil_radius * n_processing_elements;

    /**
     * \brief The width of the processed tile with the tile halo attached.
     */
    static constexpr uindex_t max_input_tile_width = 2 * halo_radius + output_tile_width;

    /**
     * \brief The height of the processed tile with the tile halo attached.
     */
    static constexpr uindex_t input_tile_height = 2 * halo_radius + output_tile_height;

    /**
     * \brief The total number of cells to read from the `in_pipe`.
     */
    static constexpr uindex_t n_input_cells = max_input_tile_width * input_tile_height;

    using index_stencil_t = typename StencilImpl::index_stencil_t;
    using uindex_stencil_t = typename StencilImpl::uindex_stencil_t;
    using StencilID = typename StencilImpl::StencilID;
    using StencilUID = typename StencilImpl::StencilUID;

    static constexpr unsigned long bits_1d =
        std::bit_width(std::max(max_input_tile_width, input_tile_height));
    using index_1d_t = ac_int<bits_1d + 1, true>;
    using uindex_1d_t = ac_int<bits_1d, false>;

    static constexpr unsigned long bits_2d = 2 * bits_1d;
    using index_2d_t = ac_int<bits_2d + 1, true>;
    using uindex_2d_t = ac_int<bits_2d, false>;

    // bits_pes must not be 1 since modulo operations on ac_int<1, false> don't work with oneAPI
    // 2022.2. If they work with a future version of oneAPI, this can be reverted.
    static constexpr unsigned long bits_pes =
        std::max<int>(2, std::bit_width(n_processing_elements));
    using index_pes_t = ac_int<bits_pes + 1, true>;
    using uindex_pes_t = ac_int<bits_pes, false>;

    /**
     * \brief Create and configure the execution kernel.
     *
     * \param trans_func The instance of the transition function to use.
     * \param i_generation The generation index of the input cells.
     * \param target_i_generation The number of generations to compute. If this number is bigger
     * than `n_processing_elements`, only `n_processing_elements` generations will be computed.
     * \param grid_c_offset The column offset of the processed tile relative to the grid's origin,
     * not including the halo. For example, for the most north-western tile the offset will always
     * be (0,0), not (-halo_radius,-halo_radius)
     * \param grid_r_offset The row offset of the processed tile relative to the grid's origin. See
     * `grid_c_offset` for details.
     * \param grid_width The number of cell columns in the grid.
     * \param grid_height The number of cell rows in the grid.
     * \param halo_value The value of cells in the grid halo.
     */
    StencilUpdateKernel(TransFunc trans_func, uindex_t i_generation, uindex_t target_i_generation,
                        uindex_t grid_c_offset, uindex_t grid_r_offset, uindex_t grid_width,
                        uindex_t grid_height, Cell halo_value)
        : trans_func(trans_func), i_generation(i_generation),
          target_i_generation(target_i_generation), grid_c_offset(grid_c_offset),
          grid_r_offset(grid_r_offset), grid_width(grid_width), grid_height(grid_height),
          halo_value(halo_value) {
        assert(grid_c_offset % output_tile_width == 0);
        assert(grid_r_offset % output_tile_height == 0);
    }

    /**
     * \brief Execute the configured operations.
     */
    void operator()() const {
        uindex_1d_t input_tile_c = 0;
        uindex_1d_t input_tile_r = 0;

        /*
         * The intel::numbanks attribute requires a power of two as it's argument and if the
         * number of processing elements isn't a power of two, it would produce an error. Therefore,
         * we calculate the next power of two and use it to allocate the cache. The compiler is
         * smart enough to see that these additional banks in the cache aren't used and therefore
         * optimizes them away.
         */
        [[intel::fpga_memory,
          intel::numbanks(2 * std::bit_ceil(n_processing_elements))]] Padded<Cell>
            cache[2][input_tile_height][std::bit_ceil(n_processing_elements)][stencil_diameter - 1];
        [[intel::fpga_register]] Cell stencil_buffer[n_processing_elements][stencil_diameter]
                                                    [stencil_diameter];

        uindex_1d_t tile_section_width = std::min(output_tile_width, grid_width - grid_c_offset);
        uindex_1d_t tile_section_height = output_tile_height;
        uindex_2d_t n_iterations =
            (tile_section_width + 2 * halo_radius) * (tile_section_height + 2 * halo_radius);

        for (uindex_2d_t i = 0; i < n_iterations; i++) {
            [[intel::fpga_register]] Cell carry = in_pipe::read();

#pragma unroll
            for (uindex_pes_t i_processing_element = 0;
                 i_processing_element < uindex_pes_t(n_processing_elements);
                 i_processing_element++) {
                /*
                 * Shift up every value in the stencil_buffer.
                 * This operation does not touch the values in the bottom row, which will be filled
                 * from the cache and the new input value later.
                 */
#pragma unroll
                for (uindex_stencil_t r = 0; r < uindex_stencil_t(stencil_diameter - 1); r++) {
#pragma unroll
                    for (uindex_stencil_t c = 0; c < uindex_stencil_t(stencil_diameter); c++) {
                        stencil_buffer[i_processing_element][c][r] =
                            stencil_buffer[i_processing_element][c][r + 1];
                    }
                }

                index_1d_t rel_input_grid_c =
                    index_1d_t(input_tile_c) -
                    index_1d_t((stencil_diameter - 1) +
                               (n_processing_elements + i_processing_element - 2) *
                                   TransFunc::stencil_radius);
                index_t input_grid_c = grid_c_offset + rel_input_grid_c.to_int64();
                index_1d_t rel_input_grid_r =
                    index_1d_t(input_tile_r) -
                    index_1d_t((stencil_diameter - 1) +
                               (n_processing_elements + i_processing_element - 2) *
                                   TransFunc::stencil_radius);
                index_t input_grid_r = grid_r_offset + rel_input_grid_r.to_int64();

                // Update the stencil buffer and cache with previous cache contents and the new
                // input cell.
#pragma unroll
                for (uindex_stencil_t cache_c = 0; cache_c < uindex_stencil_t(stencil_diameter);
                     cache_c++) {
                    Cell new_value;
                    if (cache_c == uindex_stencil_t(stencil_diameter - 1)) {
                        bool is_halo = (grid_c_offset == 0 && rel_input_grid_c < 0);
                        is_halo |= (grid_r_offset == 0 && rel_input_grid_r < 0);
                        is_halo |= input_grid_c >= grid_width || input_grid_r >= grid_height;

                        new_value = is_halo ? halo_value : carry;
                    } else {
                        new_value =
                            cache[input_tile_c[0]][input_tile_r][i_processing_element][cache_c]
                                .value;
                    }

                    stencil_buffer[i_processing_element][cache_c][stencil_diameter - 1] = new_value;
                    if (cache_c > 0) {
                        cache[(~input_tile_c)[0]][input_tile_r][i_processing_element][cache_c - 1]
                            .value = new_value;
                    }
                }

                uindex_t pe_generation =
                    (i_generation + i_processing_element / TransFunc::n_subgenerations).to_uint();
                uindex_t pe_subgeneration =
                    (i_processing_element % TransFunc::n_subgenerations).to_uint();
                index_t output_grid_c = input_grid_c - index_t(TransFunc::stencil_radius);
                index_t output_grid_r = input_grid_r - index_t(TransFunc::stencil_radius);
                TDV tdv = trans_func.get_time_dependent_value(pe_generation);
                StencilImpl stencil(ID(output_grid_c, output_grid_r), UID(grid_width, grid_height),
                                    pe_generation, pe_subgeneration,
                                    i_processing_element.to_uint64(), tdv,
                                    stencil_buffer[i_processing_element]);

                if (pe_generation < target_i_generation) {
                    carry = trans_func(stencil);
                } else {
                    carry = stencil_buffer[i_processing_element][TransFunc::stencil_radius]
                                          [TransFunc::stencil_radius];
                }
            }

            bool is_valid_output =
                input_tile_c >= uindex_1d_t((stencil_diameter - 1) * n_processing_elements);
            is_valid_output &=
                input_tile_r >= uindex_1d_t((stencil_diameter - 1) * n_processing_elements);

            if (is_valid_output) {
                out_pipe::write(carry);
            }

            if (input_tile_r == uindex_1d_t(input_tile_height - 1)) {
                input_tile_r = 0;
                input_tile_c++;
            } else {
                input_tile_r++;
            }
        }
    }

  private:
    TransFunc trans_func;
    uindex_t i_generation;
    uindex_t target_i_generation;
    uindex_t grid_c_offset;
    uindex_t grid_r_offset;
    uindex_t grid_width;
    uindex_t grid_height;
    Cell halo_value;
};

template <concepts::TransitionFunction F, uindex_t n_processing_elements = 1,
          uindex_t tile_width = 1024, uindex_t tile_height = 1024, uindex_t word_size = 64>
class StencilUpdate {
  public:
    using Cell = F::Cell;
    static constexpr uindex_t halo_radius = F::stencil_radius * n_processing_elements;
    using GridImpl = Grid<Cell, tile_width, tile_height, halo_radius, word_size>;

    struct Params {
        F transition_function;
        Cell halo_value = Cell();
        uindex_t generation_offset = 0;
        uindex_t n_generations = 1;
        sycl::device device = sycl::device();
        bool blocking = false;
        bool profiling = false;
    };

    StencilUpdate(Params params)
        : params(params), n_processed_cells(0), work_events(), walltime(0.0) {}

    Params &get_params() { return params; }

    GridImpl operator()(GridImpl &source_grid) {
        using in_pipe = sycl::pipe<class monotile_in_pipe, Cell>;
        using out_pipe = sycl::pipe<class monotile_out_pipe, Cell>;
        using ExecutionKernelImpl = StencilUpdateKernel<F, n_processing_elements, tile_width,
                                                        tile_height, in_pipe, out_pipe>;

        std::array<sycl::queue, 6> input_kernel_queues;
        for (uindex_t i = 0; i < 6; i++) {
            input_kernel_queues[i] =
                sycl::queue(params.device, {sycl::property::queue::in_order{}});
        }
        std::array<sycl::queue, 4> output_kernel_queues;
        for (uindex_t i = 0; i < 4; i++) {
            output_kernel_queues[i] =
                sycl::queue(params.device, {sycl::property::queue::in_order{}});
        }
        sycl::queue working_queue =
            sycl::queue(params.device, {cl::sycl::property::queue::enable_profiling{},
                                        sycl::property::queue::in_order{}});

        GridImpl swap_grid_a = source_grid.make_similar();
        GridImpl swap_grid_b = source_grid.make_similar();

        uindex_t gens_per_pass = ExecutionKernelImpl::gens_per_pass;
        GridImpl *pass_source = &source_grid;
        GridImpl *pass_target = &swap_grid_b;

        UID tile_range = source_grid.get_tile_range();
        uindex_t grid_width = source_grid.get_grid_width();
        uindex_t grid_height = source_grid.get_grid_height();

        auto walltime_start = std::chrono::high_resolution_clock::now();

        for (uindex_t i_gen = 0; i_gen < params.n_generations; i_gen += gens_per_pass) {
            for (uindex_t i_tile_c = 0; i_tile_c < tile_range.c; i_tile_c++) {
                for (uindex_t i_tile_r = 0; i_tile_r < tile_range.r; i_tile_r++) {
                    pass_source->template submit_read<in_pipe>(input_kernel_queues, i_tile_c,
                                                               i_tile_r);

                    auto work_event = working_queue.submit([&](sycl::handler &cgh) {
                        uindex_t c_offset = i_tile_c * tile_width;
                        uindex_t r_offset = i_tile_r * tile_height;

                        ExecutionKernelImpl exec_kernel(
                            params.transition_function, params.generation_offset + i_gen,
                            params.generation_offset + params.n_generations, c_offset, r_offset,
                            grid_width, grid_height, params.halo_value);
                        cgh.single_task<ExecutionKernelImpl>(exec_kernel);
                    });
                    if (params.profiling) {
                        work_events.push_back(work_event);
                    }

                    pass_target->template submit_write<out_pipe>(output_kernel_queues, i_tile_c,
                                                                 i_tile_r);
                }
            }

            if (i_gen == 0) {
                pass_source = &swap_grid_b;
                pass_target = &swap_grid_a;
            } else {
                std::swap(pass_source, pass_target);
            }
        }

        if (params.blocking) {
            for (sycl::queue queue : output_kernel_queues) {
                queue.wait();
            }
        }

        auto walltime_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> walltime = walltime_end - walltime_start;
        this->walltime += walltime.count();

        n_processed_cells +=
            params.n_generations * source_grid.get_grid_width() * source_grid.get_grid_height();

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

} // namespace tiling
} // namespace stencil