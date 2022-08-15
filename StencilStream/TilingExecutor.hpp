/*
 * Copyright © 2020-2022 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "SingleContextExecutor.hpp"
#include "tiling/ExecutionKernel.hpp"
#include "tiling/Grid.hpp"

#include <algorithm>

namespace stencil {

/**
 * \brief The default stencil executor.
 *
 * Unlike \ref MonotileExecutor, this executors supports any grid range by tiling the grid as
 * described in \ref tiling, at the cost of complexer IO and a computational overhead for every
 * tile.
 *
 * This executor is called `TilingExecutor` since was the first one in StencilStream 2.x. A better
 * name for it would be `TilingExecutor`.
 *
 * \tparam TransFunc The type of the transition function.
 * \tparam n_processing_elements The number of processing elements per kernel. Must be at least 1.
 * Defaults to 1.
 * \tparam tile_width The number of columns in a tile and maximum number of columns in a grid.
 * Defaults to 1024.
 * \tparam tile_height The number of rows in a tile and maximum number of rows in a grid. Defaults
 * to 1024.
 */
template <TransitionFunction TransFunc, tdv::HostState TDVS = tdv::NoneSupplier,
          uindex_t n_processing_elements = 1, uindex_t tile_width = 1024,
          uindex_t tile_height = 1024>
class TilingExecutor : public SingleContextExecutor<TransFunc, TDVS> {
  public:
    using Cell = typename TransFunc::Cell;

    /**
     * \brief The number of cells that have be added to the tile in every direction to form the
     * complete input.
     */

    static constexpr uindex_t halo_radius = TransFunc::stencil_radius * n_processing_elements;

    /**
     * \brief Create a new stencil executor.
     *
     * \param halo_value The value of cells in the grid halo.
     * \param trans_func An instance of the transition function type.
     */
    TilingExecutor(Cell halo_value, TransFunc trans_func)
        : SingleContextExecutor<TransFunc>(halo_value, trans_func),
          input_grid(cl::sycl::buffer<Cell, 2>(cl::sycl::range<2>(0, 0))) {}

    TilingExecutor(Cell halo_value, TransFunc trans_func, TDVS tdvs)
        : SingleContextExecutor<TransFunc>(halo_value, trans_func, tdvs),
          input_grid(cl::sycl::buffer<Cell, 2>(cl::sycl::range<2>(0, 0))) {}

    void set_input(cl::sycl::buffer<Cell, 2> input_buffer) override {
        this->input_grid = GridImpl(input_buffer);
    }

    void copy_output(cl::sycl::buffer<Cell, 2> output_buffer) override {
        input_grid.copy_to(output_buffer);
    }

    UID get_grid_range() const override { return input_grid.get_grid_range(); }

    void run(uindex_t n_generations) override {
        using feed_in_pipe_0 = cl::sycl::pipe<class feed_in_pipe_0_id, Cell>;
        using feed_in_pipe_1 = cl::sycl::pipe<class feed_in_pipe_1_id, Cell>;
        using feed_in_pipe_2 = cl::sycl::pipe<class feed_in_pipe_2_id, Cell>;
        using feed_in_pipe_3 = cl::sycl::pipe<class feed_in_pipe_3_id, Cell>;
        using feed_in_pipe_4 = cl::sycl::pipe<class feed_in_pipe_4_id, Cell>;
        using in_pipe = cl::sycl::pipe<class tiling_in_pipe, Cell>;
        using out_pipe = cl::sycl::pipe<class tiling_out_pipe, Cell>;
        using feed_out_pipe_0 = cl::sycl::pipe<class feed_out_pipe_0_id, Cell>;
        using feed_out_pipe_1 = cl::sycl::pipe<class feed_out_pipe_1_id, Cell>;
        using feed_out_pipe_2 = cl::sycl::pipe<class feed_out_pipe_2_id, Cell>;

        using ExecutionKernelImpl =
            tiling::ExecutionKernel<TransFunc, TDVS, n_processing_elements, tile_width, tile_height,
                                    in_pipe, out_pipe>;

        cl::sycl::queue input_queue[5] = {this->new_queue(true), this->new_queue(true),
                                          this->new_queue(true), this->new_queue(true),
                                          this->new_queue(true)};
        cl::sycl::queue merge_queue = this->new_queue(true);
        cl::sycl::queue work_queue = this->new_queue(true);
        cl::sycl::queue fork_queue = this->new_queue(true);
        cl::sycl::queue output_queue[3] = {this->new_queue(true), this->new_queue(true),
                                           this->new_queue(true)};

        uindex_t target_i_generation = this->get_i_generation() + n_generations;
        uindex_t grid_width = input_grid.get_grid_range().c;
        uindex_t grid_height = input_grid.get_grid_range().r;

        while (this->get_i_generation() < target_i_generation) {
            GridImpl output_grid = input_grid.make_output_grid();

            uindex_t delta_n_generations = std::min(target_i_generation - this->get_i_generation(),
                                                    ExecutionKernelImpl::gens_per_pass);

            cl::sycl::buffer<typename TDVS::GlobalState, 1> global_state_buffer(
                cl::sycl::range<1>(1));
            {
                auto ac = global_state_buffer
                              .template get_access<cl::sycl::access::mode::discard_write>();
                ac[0] = this->get_tdvs().prepare_global_state(this->get_i_generation(),
                                                              delta_n_generations);
            }

            std::vector<cl::sycl::event> events;
            events.reserve(input_grid.get_tile_range().c * input_grid.get_tile_range().r);

            for (uindex_t c = 0; c < input_grid.get_tile_range().c; c++) {
                uindex_t column_offset = c * tile_width;
                uindex_t n_inner_columns = std::min(tile_width, grid_width - column_offset);

                uindex_t required_columns[5];
                required_columns[0] = halo_radius;

                required_columns[1] = std::min(n_inner_columns, halo_radius);

                if (n_inner_columns > halo_radius) {
                    required_columns[2] =
                        std::min(n_inner_columns - halo_radius, TileImpl::core_width);
                } else {
                    required_columns[2] = 0;
                }

                if (n_inner_columns > halo_radius + TileImpl::core_width) {
                    required_columns[3] =
                        std::min(n_inner_columns - halo_radius - TileImpl::core_width, halo_radius);
                } else {
                    required_columns[3] = 0;
                }

                required_columns[4] = halo_radius;

                assert(required_columns[1] + required_columns[2] + required_columns[3] ==
                       n_inner_columns);

                for (uindex_t r = 0; r < input_grid.get_tile_range().r; r++) {
                    input_grid.template submit_read<class feed_in_pipe_0_id>(
                        input_queue[0], ID(c - 1, r - 1), TileImpl::Part::SOUTH_EAST_CORNER,
                        required_columns[0]);
                    input_grid.template submit_read<class feed_in_pipe_0_id>(
                        input_queue[0], ID(c, r - 1), TileImpl::Part::SOUTH_WEST_CORNER,
                        required_columns[1]);
                    input_grid.template submit_read<class feed_in_pipe_0_id>(
                        input_queue[0], ID(c, r - 1), TileImpl::Part::SOUTH_BORDER,
                        required_columns[2]);
                    input_grid.template submit_read<class feed_in_pipe_0_id>(
                        input_queue[0], ID(c, r - 1), TileImpl::Part::SOUTH_EAST_CORNER,
                        required_columns[3]);
                    input_grid.template submit_read<class feed_in_pipe_0_id>(
                        input_queue[0], ID(c + 1, r - 1), TileImpl::Part::SOUTH_WEST_CORNER,
                        required_columns[4]);

                    input_grid.template submit_read<class feed_in_pipe_1_id>(
                        input_queue[1], ID(c - 1, r), TileImpl::Part::NORTH_EAST_CORNER,
                        required_columns[0]);
                    input_grid.template submit_read<class feed_in_pipe_1_id>(
                        input_queue[1], ID(c, r), TileImpl::Part::NORTH_WEST_CORNER,
                        required_columns[1]);
                    input_grid.template submit_read<class feed_in_pipe_1_id>(
                        input_queue[1], ID(c, r), TileImpl::Part::NORTH_BORDER,
                        required_columns[2]);
                    input_grid.template submit_read<class feed_in_pipe_1_id>(
                        input_queue[1], ID(c, r), TileImpl::Part::NORTH_EAST_CORNER,
                        required_columns[3]);
                    input_grid.template submit_read<class feed_in_pipe_1_id>(
                        input_queue[1], ID(c + 1, r), TileImpl::Part::NORTH_WEST_CORNER,
                        required_columns[4]);

                    input_grid.template submit_read<class feed_in_pipe_2_id>(
                        input_queue[2], ID(c - 1, r), TileImpl::Part::EAST_BORDER,
                        required_columns[0]);
                    input_grid.template submit_read<class feed_in_pipe_2_id>(
                        input_queue[2], ID(c, r), TileImpl::Part::WEST_BORDER, required_columns[1]);
                    input_grid.template submit_read<class feed_in_pipe_2_id>(
                        input_queue[2], ID(c, r), TileImpl::Part::CORE, required_columns[2]);
                    input_grid.template submit_read<class feed_in_pipe_2_id>(
                        input_queue[2], ID(c, r), TileImpl::Part::EAST_BORDER, required_columns[3]);
                    input_grid.template submit_read<class feed_in_pipe_2_id>(
                        input_queue[2], ID(c + 1, r), TileImpl::Part::WEST_BORDER,
                        required_columns[4]);

                    input_grid.template submit_read<class feed_in_pipe_3_id>(
                        input_queue[3], ID(c - 1, r), TileImpl::Part::SOUTH_EAST_CORNER,
                        required_columns[0]);
                    input_grid.template submit_read<class feed_in_pipe_3_id>(
                        input_queue[3], ID(c, r), TileImpl::Part::SOUTH_WEST_CORNER,
                        required_columns[1]);
                    input_grid.template submit_read<class feed_in_pipe_3_id>(
                        input_queue[3], ID(c, r), TileImpl::Part::SOUTH_BORDER,
                        required_columns[2]);
                    input_grid.template submit_read<class feed_in_pipe_3_id>(
                        input_queue[3], ID(c, r), TileImpl::Part::SOUTH_EAST_CORNER,
                        required_columns[3]);
                    input_grid.template submit_read<class feed_in_pipe_3_id>(
                        input_queue[3], ID(c + 1, r), TileImpl::Part::SOUTH_WEST_CORNER,
                        required_columns[4]);

                    input_grid.template submit_read<class feed_in_pipe_4_id>(
                        input_queue[4], ID(c - 1, r + 1), TileImpl::Part::NORTH_EAST_CORNER,
                        required_columns[0]);
                    input_grid.template submit_read<class feed_in_pipe_4_id>(
                        input_queue[4], ID(c, r + 1), TileImpl::Part::NORTH_WEST_CORNER,
                        required_columns[1]);
                    input_grid.template submit_read<class feed_in_pipe_4_id>(
                        input_queue[4], ID(c, r + 1), TileImpl::Part::NORTH_BORDER,
                        required_columns[2]);
                    input_grid.template submit_read<class feed_in_pipe_4_id>(
                        input_queue[4], ID(c, r + 1), TileImpl::Part::NORTH_EAST_CORNER,
                        required_columns[3]);
                    input_grid.template submit_read<class feed_in_pipe_4_id>(
                        input_queue[4], ID(c + 1, r + 1), TileImpl::Part::NORTH_WEST_CORNER,
                        required_columns[4]);

                    merge_queue.submit([&](cl::sycl::handler &cgh) {
                        cgh.single_task<class TilingMergeKernel>([=]() {
                            constexpr unsigned long bits_width =
                                std::bit_width(2 * halo_radius + tile_width);
                            constexpr unsigned long bits_height =
                                std::bit_width(2 * halo_radius + tile_height);
                            using uindex_width_t = ac_int<bits_width, false>;
                            using uindex_height_t = ac_int<bits_height, false>;

                            [[intel::loop_coalesce(2)]] for (uindex_width_t c = 0;
                                                             c < uindex_width_t(2 * halo_radius +
                                                                                n_inner_columns);
                                                             c++) {
                                for (uindex_height_t r = 0;
                                     r < uindex_height_t(2 * halo_radius + tile_height); r++) {
                                    Cell value;
                                    if (r < uindex_height_t(halo_radius)) {
                                        value = feed_in_pipe_0::read();
                                    } else if (r < uindex_height_t(2 * halo_radius)) {
                                        value = feed_in_pipe_1::read();
                                    } else if (r < uindex_height_t(2 * halo_radius +
                                                                   GridImpl::core_height)) {
                                        value = feed_in_pipe_2::read();
                                    } else if (r < uindex_height_t(3 * halo_radius +
                                                                   GridImpl::core_height)) {
                                        value = feed_in_pipe_3::read();
                                    } else {
                                        value = feed_in_pipe_4::read();
                                    }
                                    in_pipe::write(value);
                                }
                            }
                        });
                    });

                    cl::sycl::event computation_event =
                        work_queue.submit([&](cl::sycl::handler &cgh) {
                            auto global_state_ac =
                                global_state_buffer
                                    .template get_access<cl::sycl::access::mode::read>(cgh);

                            cgh.single_task<class TilingExecutionKernel>(ExecutionKernelImpl(
                                this->get_trans_func(), this->get_i_generation(),
                                target_i_generation, c * tile_width, r * tile_height, grid_width,
                                grid_height, this->get_halo_value(), global_state_ac));
                        });
                    events.push_back(computation_event);

                    fork_queue.submit([&](cl::sycl::handler &cgh) {
                        cgh.single_task<class TilingForkKernel>([=]() {
                            constexpr unsigned long bits_width = std::bit_width(tile_width);
                            constexpr unsigned long bits_height = std::bit_width(tile_height);
                            using uindex_width_t = ac_int<bits_width, false>;
                            using uindex_height_t = ac_int<bits_height, false>;

                            [[intel::loop_coalesce(2)]] for (uindex_width_t c = 0;
                                                             c < uindex_width_t(n_inner_columns);
                                                             c++) {
                                for (uindex_height_t r = 0; r < uindex_height_t(tile_height); r++) {
                                    Cell value = out_pipe::read();
                                    if (r < uindex_height_t(halo_radius)) {
                                        feed_out_pipe_0::write(value);
                                    } else if (r < uindex_height_t(tile_height - halo_radius)) {
                                        feed_out_pipe_1::write(value);
                                    } else {
                                        feed_out_pipe_2::write(value);
                                    }
                                }
                            }
                        });
                    });

                    output_grid.template submit_write<class feed_out_pipe_0_id>(
                        output_queue[0], ID(c, r), TileImpl::Part::NORTH_WEST_CORNER,
                        required_columns[1]);
                    output_grid.template submit_write<class feed_out_pipe_0_id>(
                        output_queue[0], ID(c, r), TileImpl::Part::NORTH_BORDER,
                        required_columns[2]);
                    output_grid.template submit_write<class feed_out_pipe_0_id>(
                        output_queue[0], ID(c, r), TileImpl::Part::NORTH_EAST_CORNER,
                        required_columns[3]);

                    output_grid.template submit_write<class feed_out_pipe_1_id>(
                        output_queue[1], ID(c, r), TileImpl::Part::WEST_BORDER,
                        required_columns[1]);
                    output_grid.template submit_write<class feed_out_pipe_1_id>(
                        output_queue[1], ID(c, r), TileImpl::Part::CORE, required_columns[2]);
                    output_grid.template submit_write<class feed_out_pipe_1_id>(
                        output_queue[1], ID(c, r), TileImpl::Part::EAST_BORDER,
                        required_columns[3]);

                    output_grid.template submit_write<class feed_out_pipe_2_id>(
                        output_queue[2], ID(c, r), TileImpl::Part::SOUTH_WEST_CORNER,
                        required_columns[1]);
                    output_grid.template submit_write<class feed_out_pipe_2_id>(
                        output_queue[2], ID(c, r), TileImpl::Part::SOUTH_BORDER,
                        required_columns[2]);
                    output_grid.template submit_write<class feed_out_pipe_2_id>(
                        output_queue[2], ID(c, r), TileImpl::Part::SOUTH_EAST_CORNER,
                        required_columns[3]);
                }
            }

            input_grid = output_grid;

            double earliest_start = std::numeric_limits<double>::max();
            double latest_end = std::numeric_limits<double>::min();

            for (cl::sycl::event event : events) {
                earliest_start = std::min(earliest_start, RuntimeSample::start_of_event(event));
                latest_end = std::max(latest_end, RuntimeSample::end_of_event(event));
            }
            this->get_runtime_sample().add_pass(latest_end - earliest_start);

            this->inc_i_generation(delta_n_generations);
        }
    }

  private:
    using GridImpl = tiling::Grid<Cell, tile_width, tile_height, halo_radius>;
    using TileImpl = typename GridImpl::Tile;
    GridImpl input_grid;
};

} // namespace stencil