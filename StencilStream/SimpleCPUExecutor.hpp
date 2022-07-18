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
#include "Stencil.hpp"

namespace stencil {
template <typename TransFunc> class SimpleCPUExecutor : public SingleContextExecutor<TransFunc> {
  public:
    using Cell = typename TransFunc::Cell;

    SimpleCPUExecutor(Cell halo_value, TransFunc trans_func)
        : SingleContextExecutor<TransFunc>(halo_value, trans_func), grid(cl::sycl::range<2>(1, 1)) {
        this->select_cpu();
    }

    virtual void run(uindex_t n_generations) override {
        cl::sycl::queue queue = this->new_queue();
        cl::sycl::buffer<Cell, 2> in_buffer = grid;
        cl::sycl::buffer<Cell, 2> out_buffer(grid.get_range());

        for (uindex_t i_generation = 0; i_generation < n_generations; i_generation++) {
            cl::sycl::event event = queue.submit([&](cl::sycl::handler &cgh) {
                auto in_ac = in_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
                auto out_ac =
                    out_buffer.template get_access<cl::sycl::access::mode::discard_write>(cgh);
                uindex_t gen = this->get_i_generation() + i_generation;
                uindex_t grid_width = in_ac.get_range()[0];
                uindex_t grid_height = in_ac.get_range()[1];
                Cell halo_value = this->get_halo_value();
                TransFunc trans_func = this->get_trans_func();

                cgh.parallel_for<class SimpleCPUExecutionKernel>(
                    in_ac.get_range(), [=](cl::sycl::id<2> idx) {
                        Stencil<TransFunc> stencil(idx, in_ac.get_range(), gen, i_generation);

                        for (index_t delta_c = -TransFunc::stencil_radius;
                             delta_c <= index_t(TransFunc::stencil_radius); delta_c++) {
                            for (index_t delta_r = -TransFunc::stencil_radius;
                                 delta_r <= index_t(TransFunc::stencil_radius); delta_r++) {
                                index_t c = index_t(idx[0]) + delta_c;
                                index_t r = index_t(idx[1]) + delta_r;
                                if (c < 0 || r < 0 || c >= grid_width || r >= grid_height) {
                                    stencil[ID(delta_c, delta_r)] = halo_value;
                                } else {
                                    stencil[ID(delta_c, delta_r)] = in_ac[c][r];
                                }
                            }
                        }

                        out_ac[idx] = trans_func(stencil);
                    });
            });
            this->get_runtime_sample().add_pass(event);
            std::swap(in_buffer, out_buffer);
        }

        queue.wait_and_throw();
        grid = in_buffer;
        this->inc_i_generation(n_generations);
    }

    virtual void set_input(cl::sycl::buffer<Cell, 2> input_buffer) override {
        grid = input_buffer.get_range();

        auto in_ac = input_buffer.template get_access<cl::sycl::access::mode::read>();
        auto out_ac = grid.template get_access<cl::sycl::access::mode::discard_write>();

        for (uindex_t c = 0; c < input_buffer.get_range()[0]; c++) {
            for (uindex_t r = 0; r < input_buffer.get_range()[1]; r++) {
                out_ac[c][r] = in_ac[c][r];
            }
        }
    }

    virtual void copy_output(cl::sycl::buffer<Cell, 2> output_buffer) override {
        if (grid.get_range() != output_buffer.get_range()) {
            throw std::range_error(
                "The given output buffer doesn't have the same range as the grid.");
        }

        auto in_ac = grid.template get_access<cl::sycl::access::mode::read>();
        auto out_ac = output_buffer.template get_access<cl::sycl::access::mode::discard_write>();

        for (uindex_t c = 0; c < output_buffer.get_range()[0]; c++) {
            for (uindex_t r = 0; r < output_buffer.get_range()[1]; r++) {
                out_ac[c][r] = in_ac[c][r];
            }
        }
    }

    virtual UID get_grid_range() const override {
        return UID(grid.get_range()[0], grid.get_range()[1]);
    }

  private:
    cl::sycl::buffer<Cell, 2> grid;
};
} // namespace stencil