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
          uindex_t tile_width = 1024, uindex_t tile_height = 1024>
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

    /**
     * \brief Create a new executor.
     *
     * \param halo_value The value of cells in the grid halo.
     * \param trans_func An instance of the transition function type.
     */
    [[deprecated("Use MonotileExecutor(T) instead")]]
    MonotileExecutor(T halo_value, TransFunc trans_func)
        : Parent(halo_value, trans_func), tile_buffer(cl::sycl::range<1>(tile_width * tile_height)), grid_range(1, 1) {
        auto ac = tile_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        ac[0] = halo_value;
    }

    MonotileExecutor(T halo_value) : Parent(halo_value), tile_buffer(cl::sycl::range<1>(tile_width * tile_height)), grid_range(1, 1) {
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
                if (c < grid_range.c && r < grid_range.r) {
                    tile_ac[c * tile_height + r] = in_ac[c][r];
                } else {
                    tile_ac[c * tile_height + r] = this->get_halo_value();
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
                out_ac[c][r] = in_ac[c*tile_height + r];
            }
        }
    }

    UID get_grid_range() const override {
        return this->grid_range;
    }

    [[deprecated("Use run(uindex_t, std::function<TransFunc(cl::sycl::handler &)>) instead")]]
    void run(uindex_t n_generations) override {
        run(
            n_generations,
            std::function([&](cl::sycl::handler &cgh) {return this->get_trans_func();})
        );
    }

    void run(uindex_t n_generations, TransFunc trans_func) override {
        run(n_generations, std::function([=](cl::sycl::handler &cgh) { return trans_func; }));
    }

    void run(uindex_t n_generations, std::function<TransFunc(cl::sycl::handler &)> trans_func_builder) override {
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
        cl::sycl::buffer<T, 1> write_buffer = cl::sycl::range<1>(tile_width * tile_height);

        while (this->get_i_generation() < target_i_generation) {
            input_queue.submit([&](cl::sycl::handler &cgh) {
                auto ac = read_buffer.template get_access<cl::sycl::access::mode::read>(cgh);

                cgh.single_task<class MonotileInputKernel>([=]() {
                    for (uindex_t i = 0; i < tile_width * tile_height; i++) {
                        in_pipe::write(ac[i]);
                    }
                });
            });

            cl::sycl::event computation_event = work_queue.submit([&](cl::sycl::handler &cgh) {
                TransFunc trans_func = trans_func_builder(cgh);
                cgh.single_task<class MonotileExecutionKernel>(ExecutionKernelImpl(
                    trans_func, this->get_i_generation(), target_i_generation,
                    grid_width, grid_height, this->get_halo_value()));
            });

            output_queue.submit([&](cl::sycl::handler &cgh) {
                auto ac =
                    write_buffer.template get_access<cl::sycl::access::mode::discard_write>(cgh);

                cgh.single_task<class MonotileOutputKernel>([=]() {
                    for (uindex_t i = 0; i < tile_width * tile_height; i++) {
                        ac[i] = out_pipe::read();
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