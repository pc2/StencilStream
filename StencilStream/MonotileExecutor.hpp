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
#include "MonotileExecutionKernel.hpp"
#include "SingleQueueExecutor.hpp"

namespace stencil {
template <typename T, uindex_t stencil_radius, typename TransFunc, uindex_t pipeline_length = 1,
          uindex_t tile_width = 1024, uindex_t tile_height = 1024, uindex_t burst_size = 1024>
class MonotileExecutor : public SingleQueueExecutor<T, stencil_radius, TransFunc, pipeline_length> {
  public:
    static constexpr uindex_t burst_length = std::min<uindex_t>(1, burst_size / sizeof(T));
    static constexpr uindex_t halo_radius = stencil_radius * pipeline_length;
    using Parent = SingleQueueExecutor<T, stencil_radius, TransFunc, pipeline_length>;

    MonotileExecutor(T halo_value, TransFunc trans_func)
        : Parent(halo_value, trans_func), tile_buffer(cl::sycl::range<2>(tile_width, tile_height)) {
        auto ac = tile_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        ac[0][0] = halo_value;
    }

    void set_input(cl::sycl::buffer<T, 2> input_buffer) override {
        if (input_buffer.get_range()[0] > tile_width && input_buffer.get_range()[1] > tile_height) {
            throw std::range_error("The grid is bigger than the tile. The monotile architecture "
                                   "requires that grid ranges are smaller or equal to the tile "
                                   "range");
        }
        auto in_ac = input_buffer.template get_access<cl::sycl::access::mode::read>();
        tile_buffer = cl::sycl::buffer<T, 2>(input_buffer.get_range());
        auto tile_ac = tile_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        for (uindex_t c = 0; c < input_buffer.get_range()[0]; c++) {
            for (uindex_t r = 0; r < input_buffer.get_range()[1]; r++) {
                tile_ac[c][r] = in_ac[c][r];
            }
        }
    }

    void copy_output(cl::sycl::buffer<T, 2> output_buffer) override {
        if (output_buffer.get_range() != tile_buffer.get_range()) {
            throw std::range_error("The output buffer is not the same size as the grid");
        }
        auto in_ac = tile_buffer.template get_access<cl::sycl::access::mode::read>();
        auto out_ac = output_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        for (uindex_t c = 0; c < tile_buffer.get_range()[0]; c++) {
            for (uindex_t r = 0; r < tile_buffer.get_range()[1]; r++) {
                out_ac[c][r] = in_ac[c][r];
            }
        }
    }

    UID get_grid_range() const override {
        return UID(tile_buffer.get_range()[0], tile_buffer.get_range()[1]);
    }

  protected:
    std::optional<double> run_pass(uindex_t target_i_generation) override {
        using in_pipe = cl::sycl::pipe<class monotile_in_pipe, T>;
        using out_pipe = cl::sycl::pipe<class monotile_out_pipe, T>;
        using ExecutionKernelImpl =
            MonotileExecutionKernel<TransFunc, T, stencil_radius, pipeline_length, tile_width,
                                    tile_height, in_pipe, out_pipe>;

        cl::sycl::queue &queue = this->get_queue();

        uindex_t grid_width = tile_buffer.get_range()[0];
        uindex_t grid_height = tile_buffer.get_range()[1];
        cl::sycl::buffer<T, 2> out_buffer(tile_buffer.get_range());

        queue.submit([&](cl::sycl::handler &cgh) {
            auto ac = tile_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
            T halo_value = this->get_halo_value();

            cgh.single_task<class MonotileInputKernel>([=]() {
                [[intel::loop_coalesce(2)]] for (uindex_t c = 0; c < tile_width; c++) {
                    for (uindex_t r = 0; r < tile_height; r++) {
                        T value;
                        if (c < grid_width && r < grid_height) {
                            value = ac[c][r];
                        } else {
                            value = halo_value;
                        }

                        in_pipe::write(value);
                    }
                }
            });
        });

        cl::sycl::event computation_event = queue.submit([&](cl::sycl::handler &cgh) {
            cgh.single_task(ExecutionKernelImpl(this->get_trans_func(), this->get_i_generation(),
                                                target_i_generation, grid_width, grid_height,
                                                this->get_halo_value()));
        });

        queue.submit([&](cl::sycl::handler &cgh) {
            auto ac = out_buffer.template get_access<cl::sycl::access::mode::discard_write>(cgh);
            T halo_value = this->get_halo_value();

            cgh.single_task<class MonotileInputKernel>([=]() {
                [[intel::loop_coalesce(2)]] for (uindex_t c = 0; c < tile_width; c++) {
                    for (uindex_t r = 0; r < tile_height; r++) {
                        T value = out_pipe::read();
                        if (c < grid_width && r < grid_height) {
                            ac[c][r] = value;
                        }
                    }
                }
            });
        });

        tile_buffer = out_buffer;

        if (this->is_runtime_analysis_enabled()) {
            return RuntimeSample::runtime_of_event(computation_event);
        } else {
            return std::nullopt;
        }
    }

  private:
    cl::sycl::buffer<T, 2> tile_buffer;
    UID grid_range;
};
} // namespace stencil