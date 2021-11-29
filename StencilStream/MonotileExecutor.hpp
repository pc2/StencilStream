/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel
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
#include "SingleContextExecutor.hpp"
#include "monotile/ExecutionKernel.hpp"

namespace stencil {
template <typename T, uindex_t stencil_radius, typename TransFunc, uindex_t pipeline_length = 1,
          uindex_t tile_width = 1024, uindex_t tile_height = 1024, uindex_t burst_size=64>
/**
 * \brief An executor that follows \ref monotile.
 *
 * The feature that distincts this executor from \ref StencilExecutor is that it works with exactly
 * one tile. This means the grid range may not exceed the set tile range, but it uses less resources
 * and time per kernel execution.
 *
 * \tparam T The cell type.
 * \tparam stencil_radius The radius of the stencil buffer supplied to the transition function.
 * \tparam TransFunc The type of the transition function.
 * \tparam pipeline_length The number of hardware execution stages per kernel. Must be at least 1.
 * Defaults to 1.
 * \tparam tile_width The number of columns in a tile and maximum number of columns in a grid.
 * Defaults to 1024.
 * \tparam tile_height The number of rows in a tile and maximum number of rows in a grid. Defaults
 * to 1024.
 */
class MonotileExecutor : public SingleContextExecutor<T, stencil_radius, TransFunc> {
  public:
    /**
     * \brief Shorthand for the parent class.
     */
    using Parent = SingleContextExecutor<T, stencil_radius, TransFunc>;

    static constexpr uindex_t burst_length = sizeof(T) / burst_size + (sizeof(T) % burst_size == 0 ? 0 : 1);
    static constexpr uindex_t n_bursts = tile_width * tile_height / burst_length + (tile_width * tile_height % burst_length == 0 ? 0 : 1);

    /**
     * \brief Create a new executor.
     *
     * \param halo_value The value of cells in the grid halo.
     * \param trans_func An instance of the transition function type.
     */
    MonotileExecutor(T halo_value, TransFunc trans_func)
        : Parent(halo_value, trans_func), tile_buffer(cl::sycl::range<1>(n_bursts * burst_length)), grid_range(1, 1) {
        auto ac = tile_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        ac[0] = halo_value;
    }

    /**
     * \brief Set the internal state of the grid.
     *
     * This will copy the contents of the buffer to an internal representation. The buffer may be
     * used for other purposes later. It must not reset the generation index. The range of the input
     * buffer will be used as the new grid range.
     *
     * \throws std::range_error Thrown if the number of width or height of the buffer exceeds the
     * set width and height of the tile. \param input_buffer The source buffer of the new grid
     * state.
     */
    void set_input(cl::sycl::buffer<T, 2> input_buffer) override {
        if (input_buffer.get_range()[0] > tile_width && input_buffer.get_range()[1] > tile_height) {
            throw std::range_error("The grid is bigger than the tile. The monotile architecture "
                                   "requires that grid ranges are smaller or equal to the tile "
                                   "range");
        }
        grid_range.c = input_buffer.get_range()[0];
        grid_range.r = input_buffer.get_range()[1];

        auto in_ac = input_buffer.template get_access<cl::sycl::access::mode::read>();
        auto tile_ac = tile_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        for (uindex_t c = 0; c < tile_width; c++) {
            for (uindex_t r = 0; r < tile_height; r++) {
                uindex_t burst_i = (c * tile_width + r) / burst_length;
                uindex_t cell_i = (c * tile_width + r) % burst_length;
                if (c < grid_range.c && r < grid_range.r) {
                    tile_ac[burst_i * burst_length + cell_i] = in_ac[c][r];
                } else {
                    tile_ac[burst_i * burst_length + cell_i] = this->get_halo_value();
                }
            }
        }
    }

    void copy_output(cl::sycl::buffer<T, 2> output_buffer) override {
        if (output_buffer.get_range()[0] != grid_range.c || output_buffer.get_range()[1] != grid_range.r) {
            throw std::range_error("The output buffer is not the same size as the grid");
        }
        auto in_ac = tile_buffer.template get_access<cl::sycl::access::mode::read>();
        auto out_ac = output_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        for (uindex_t c = 0; c < output_buffer.get_range()[0]; c++) {
            for (uindex_t r = 0; r < output_buffer.get_range()[1]; r++) {
                uindex_t burst_i = (c * tile_width + r) / burst_length;
                uindex_t cell_i = (c * tile_width + r) % burst_length;
                out_ac[c][r] = in_ac[burst_i * burst_length + cell_i];
            }
        }
    }

    UID get_grid_range() const override {
        return this->grid_range;
    }

    void run(uindex_t n_generations) override {
        using in_pipe = cl::sycl::pipe<class monotile_in_pipe, T>;
        using out_pipe = cl::sycl::pipe<class monotile_out_pipe, T>;
        using ExecutionKernelImpl =
            monotile::ExecutionKernel<TransFunc, T, stencil_radius, pipeline_length, tile_width,
                                      tile_height, in_pipe, out_pipe>;

        cl::sycl::queue input_queue = this->new_queue(true);
        cl::sycl::queue work_queue = this->new_queue(true);
        cl::sycl::queue output_queue = this->new_queue(true);

        uindex_t target_i_generation = this->get_i_generation() + n_generations;
        uindex_t grid_width = grid_range.c;
        uindex_t grid_height = grid_range.r;

        cl::sycl::buffer<T, 1> read_buffer = tile_buffer;
        cl::sycl::buffer<T, 1> write_buffer = cl::sycl::range<1>(n_bursts * burst_length);

        while (this->get_i_generation() < target_i_generation) {
            input_queue.submit([&](cl::sycl::handler &cgh) {
                auto ac = read_buffer.template get_access<cl::sycl::access::mode::read>(cgh);

                cgh.single_task<class MonotileInputKernel>([=]() {
                    [[intel::loop_coalesce]]
                    for (uindex_t burst_i = 0; burst_i * burst_length < tile_width * tile_height; burst_i++) {
                        T cache[burst_length];

                        #pragma unroll
                        for (uindex_t cell_i = 0; cell_i < burst_length; cell_i++) {
                            cache[cell_i] = ac[burst_i * burst_length + cell_i];
                        }

                        for (uindex_t cell_i = 0; cell_i < burst_length; cell_i++) {
                            if (burst_i * burst_length + cell_i < tile_width * tile_height) {
                                in_pipe::write(cache[cell_i]);
                            }
                        }
                    }
                });
            });

            cl::sycl::event computation_event = work_queue.submit([&](cl::sycl::handler &cgh) {
                cgh.single_task<class MonotileExecutionKernel>(ExecutionKernelImpl(
                    this->get_trans_func(), this->get_i_generation(), target_i_generation,
                    grid_width, grid_height, this->get_halo_value()));
            });

            output_queue.submit([&](cl::sycl::handler &cgh) {
                auto ac =
                    write_buffer.template get_access<cl::sycl::access::mode::discard_write>(cgh);

                cgh.single_task<class MonotileOutputKernel>([=]() {
                    [[intel::loop_coalesce]]
                    for (uindex_t burst_i = 0; burst_i * burst_length < tile_width * tile_height; burst_i++) {
                        T cache[burst_length];

                        for (uindex_t cell_i = 0; cell_i < burst_length; cell_i++) {
                            if (burst_i * burst_length + cell_i < tile_width * tile_height) {
                                cache[cell_i] = out_pipe::read();
                            }
                        }

                        #pragma unroll
                        for (uindex_t cell_i = 0; cell_i < burst_length; cell_i++) {
                            ac[burst_i * burst_length + cell_i] = cache[cell_i]; 
                        }
                    }
                });
            });

            std::swap(read_buffer, write_buffer);

            this->get_runtime_sample().add_pass(computation_event);

            this->inc_i_generation(
                std::min(target_i_generation - this->get_i_generation(), pipeline_length));
        }

        tile_buffer = read_buffer;
    }

  private:
    cl::sycl::buffer<T, 1> tile_buffer;
    UID grid_range;
};
} // namespace stencil