/*
 * Copyright © 2020-2022 Jan-Oliver Opdenhövel, Paderborn Center for Parallel
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
#include <boost/preprocessor/cat.hpp>

#include <numeric>

namespace stencil {

template<TransitionFunction TransFunc>
class MonotileInputKernel;


template<TransitionFunction TransFunc>
class MonotileOutputKernel;

template <TransitionFunction TransFunc, tdv::HostState TDVS, uindex_t n_processing_elements = 1,
          uindex_t tile_width = 1024, uindex_t tile_height = 1024, uindex_t word_size = 64>
/**
 * \brief An executor that follows \ref monotile.
 *
 * The feature that distincts this executor from \ref StencilExecutor is that it works with exactly
 * one tile. This means the grid range may not exceed the set tile range, but it uses less resources
 * and time per kernel execution.
 *
 * \tparam TransFunc The type of the transition function.
 * \tparam n_processing_elements The number of processing elements per kernel. Must be at least 1.
 * Defaults to 1.
 * \tparam tile_width The number of columns in a tile and maximum number of columns in a grid.
 * Defaults to 1024.
 * \tparam tile_height The number of rows in a tile and maximum number of rows in a grid. Defaults
 * to 1024.
 */
class MonotileExecutor : public SingleContextExecutor<TransFunc, TDVS> {
  public:
    using Cell = typename TransFunc::Cell;

    static constexpr uindex_t word_length =
        std::lcm(sizeof(Padded<Cell>), word_size) / sizeof(Padded<Cell>);
    static constexpr uindex_t n_words = n_cells_to_n_words(tile_width * tile_height, word_length);

    using IOWord = std::array<Padded<Cell>, word_length>;

    static constexpr unsigned long bits_cell = std::bit_width(word_length);
    using index_cell_t = ac_int<bits_cell + 1, true>;
    using uindex_cell_t = ac_int<bits_cell, false>;

    static constexpr unsigned long bits_word = std::bit_width(n_words);
    using index_word_t = ac_int<bits_word + 1, true>;
    using uindex_word_t = ac_int<bits_word, false>;

    /**
     * \brief Create a new executor.
     *
     * \param trans_func An instance of the transition function type.
     */
    MonotileExecutor(Cell halo_value, TransFunc trans_func)
        : SingleContextExecutor<TransFunc, TDVS>(halo_value, trans_func),
          tile_buffer(cl::sycl::range<1>(1)), grid_range(1, 1) {
        auto ac = tile_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        ac[0][0].value = this->get_halo_value();
    }

    MonotileExecutor(Cell halo_value, TransFunc trans_func, TDVS tdvs)
        : SingleContextExecutor<TransFunc, TDVS>(halo_value, trans_func, tdvs),
          tile_buffer(cl::sycl::range<1>(1)), grid_range(1, 1) {
        auto ac = tile_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        ac[0][0].value = this->get_halo_value();
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
    void set_input(cl::sycl::buffer<Cell, 2> input_buffer) override {
        if (input_buffer.get_range()[0] > tile_width || input_buffer.get_range()[1] > tile_height) {
            throw std::range_error("The grid is bigger than the tile. The monotile architecture "
                                   "requires that grid ranges are smaller or equal to the tile "
                                   "range");
        }
        grid_range.c = input_buffer.get_range()[0];
        grid_range.r = input_buffer.get_range()[1];
        tile_buffer =
            cl::sycl::range<1>(n_cells_to_n_words(grid_range.c * grid_range.r, word_length));

        auto in_ac = input_buffer.template get_access<cl::sycl::access::mode::read>();
        auto tile_ac = tile_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        for (uindex_t c = 0; c < grid_range.c; c++) {
            for (uindex_t r = 0; r < grid_range.r; r++) {
                uindex_t word_i = (c * grid_range.r + r) / word_length;
                uindex_t cell_i = (c * grid_range.r + r) % word_length;
                tile_ac[word_i][cell_i].value = in_ac[c][r];
            }
        }
    }

    void copy_output(cl::sycl::buffer<Cell, 2> output_buffer) override {
        if (output_buffer.get_range()[0] != grid_range.c ||
            output_buffer.get_range()[1] != grid_range.r) {
            throw std::range_error("The output buffer is not the same size as the grid");
        }
        auto in_ac = tile_buffer.template get_access<cl::sycl::access::mode::read>();
        auto out_ac = output_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        for (uindex_t c = 0; c < output_buffer.get_range()[0]; c++) {
            for (uindex_t r = 0; r < output_buffer.get_range()[1]; r++) {
                uindex_t word_i = (c * grid_range.r + r) / word_length;
                uindex_t cell_i = (c * grid_range.r + r) % word_length;
                out_ac[c][r] = in_ac[word_i][cell_i].value;
            }
        }
    }

    UID get_grid_range() const override { return this->grid_range; }

    void run(uindex_t n_generations) override {
        using in_pipe = cl::sycl::pipe<class monotile_in_pipe, Cell>;
        using out_pipe = cl::sycl::pipe<class monotile_out_pipe, Cell>;
        using ExecutionKernelImpl =
            monotile::ExecutionKernel<TransFunc, typename TDVS::KernelArgument, n_processing_elements,
                                      tile_width, tile_height, in_pipe, out_pipe>;
        using uindex_2d_t = typename ExecutionKernelImpl::uindex_2d_t;

        this->get_tdvs().prepare_range(this->get_i_generation(), n_generations);

        cl::sycl::queue input_queue = this->new_queue(true);
        cl::sycl::queue work_queue = this->new_queue(true);
        cl::sycl::queue output_queue = this->new_queue(true);

        uindex_t target_i_generation = this->get_i_generation() + n_generations;
        uindex_t grid_width = grid_range.c;
        uindex_t grid_height = grid_range.r;

        cl::sycl::buffer<IOWord, 1> read_buffer = tile_buffer;
        cl::sycl::buffer<IOWord, 1> write_buffer = cl::sycl::range<1>(n_words);

        while (this->get_i_generation() < target_i_generation) {
            uindex_t delta_n_generations = std::min(target_i_generation - this->get_i_generation(),
                                                    uindex_t(ExecutionKernelImpl::gens_per_pass));

            input_queue.submit([&](cl::sycl::handler &cgh) {
                auto ac = read_buffer.template get_access<cl::sycl::access::mode::read>(cgh);

                cgh.single_task([=]() {
                    [[intel::fpga_memory]] IOWord cache;

                    uindex_word_t word_i = 0;
                    uindex_cell_t cell_i = word_length;
                    for (uindex_2d_t i = 0; i < uindex_2d_t(grid_width * grid_height); i++) {
                        if (cell_i == uindex_cell_t(word_length)) {
                            cache = ac[word_i.to_uint64()];
                            word_i++;
                            cell_i = 0;
                        }
                        in_pipe::write(cache[cell_i].value);
                        cell_i++;
                    }
                });
            });

            cl::sycl::event computation_event = work_queue.submit([&](cl::sycl::handler &cgh) {
                auto tdv_global_state = this->get_tdvs().build_kernel_argument(
                    cgh, this->get_i_generation(), delta_n_generations);

                cgh.single_task(ExecutionKernelImpl(
                    this->get_trans_func(), this->get_i_generation(), target_i_generation,
                    grid_width, grid_height, this->get_halo_value(), tdv_global_state));
            });

            output_queue.submit([&](cl::sycl::handler &cgh) {
                auto ac =
                    write_buffer.template get_access<cl::sycl::access::mode::discard_write>(cgh);

                cgh.single_task([=]() {
                    [[intel::fpga_memory]] IOWord cache;

                    uindex_word_t word_i = 0;
                    uindex_cell_t cell_i = 0;
                    for (uindex_2d_t i = 0; i < uindex_2d_t(grid_width * grid_height); i++) {
                        cache[cell_i].value = out_pipe::read();
                        cell_i++;
                        if (cell_i == uindex_cell_t(word_length)) {
                            ac[word_i.to_uint64()] = cache;
                            cell_i = 0;
                            word_i++;
                        }
                    }

                    if (cell_i != 0) {
                        ac[word_i.to_uint64()] = cache;
                    }
                });
            });

            std::swap(read_buffer, write_buffer);

            this->get_runtime_sample().add_pass(computation_event);

            this->inc_i_generation(delta_n_generations);
        }

        tile_buffer = read_buffer;
    }

  private:
    cl::sycl::buffer<IOWord, 1> tile_buffer;
    UID grid_range;
};

} // namespace stencil