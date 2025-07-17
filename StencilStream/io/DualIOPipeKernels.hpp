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
#include "../Helpers.hpp"
#include "Base.hpp"

namespace stencil {
namespace io {

template <typename T, typename lower_pipe_id, typename upper_pipe_id, typename out_pipe>
class DualIOPipeRecvKernel {
    static constexpr std::size_t max_group_size = 1024;
    static constexpr std::size_t max_group_length_in_values = max_group_size / sizeof(T);
    static constexpr std::size_t max_group_length_in_pipewords = max_group_size / pipeword_size;

    using lower_recv_io_pipe =
        sycl::ext::intel::kernel_readable_io_pipe<lower_pipe_id, pipeword_t,
                                                  max_group_length_in_pipewords>;
    using upper_recv_io_pipe =
        sycl::ext::intel::kernel_readable_io_pipe<upper_pipe_id, pipeword_t,
                                                  max_group_length_in_pipewords>;

  public:
    DualIOPipeRecvKernel(std::size_t n_values) : n_values(n_values) {}

    void operator()() const
        requires(sizeof(T) == 32)
    {
        for (size_t i = 0; i < n_values; i++) {
            pipeword_t pipeword =
                (i % 2 == 0) ? lower_recv_io_pipe::read() : upper_recv_io_pipe::read();
            T value = *((T *)&pipeword);
            out_pipe::write(value);
        }
    }

    void operator()() const
        requires(sizeof(T) == 64)
    {
        for (size_t i = 0; i < n_values; i++) {
            pipeword_t pipewords[2];
            pipewords[0] = lower_recv_io_pipe::read();
            pipewords[1] = upper_recv_io_pipe::read();
            out_pipe::write(*((T *)pipewords));
        }
    }

    void operator()() const
        requires(sizeof(T) != 32 && sizeof(T) != 64 && sizeof(T) <= max_group_size &&
                 max_group_size % sizeof(T) == 0)
    {
        std::size_t n_groups = int_ceil_div(n_values, max_group_length_in_values);
        for (size_t i_group = 0; i_group < n_groups; i_group++) {
            std::size_t group_length_in_values = std::min(
                max_group_length_in_values, n_values - i_group * max_group_length_in_values);
            std::size_t group_length_in_pipewords =
                int_ceil_div(group_length_in_values * sizeof(T), pipeword_size);

            T group_buffer[max_group_length_in_values];
            pipeword_t *pipeword = (pipeword_t *)group_buffer;

            for (size_t i_word = 0; i_word < group_length_in_pipewords; i_word += 2) {
                pipeword[i_word] = lower_recv_io_pipe::read();
                if (i_word + 1 < group_length_in_pipewords) {
                    pipeword[i_word + 1] = upper_recv_io_pipe::read();
                }
            }

            for (size_t i_value = 0; i_value < group_length_in_values; i_value++) {
                out_pipe::write(group_buffer[i_value]);
            }
        }
    }

  private:
    std::size_t n_values;
};

} // namespace io
} // namespace stencil