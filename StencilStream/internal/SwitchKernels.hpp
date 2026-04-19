/*
 * Copyright © 2020-2026 Paderborn Center for Parallel Computing, Paderborn
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
#include "Helpers.hpp"
#include <bit>
#include <exception>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/sycl.hpp>

namespace stencil {
namespace internal {

template <typename T, typename in_pipe, typename low_out_pipe, typename high_out_pipe,
          std::size_t max_n_words = std::numeric_limits<std::size_t>::max()>
class ForkSwitchKernel {
  public:
    using uindex_word_t = ac_int<std::bit_width(max_n_words), false>;

    ForkSwitchKernel(std::size_t n_words, bool low_pipe_active)
        : n_words(n_words), low_pipe_active(low_pipe_active) {
        if (n_words > max_n_words) {
            throw std::out_of_range("The number of requested pipe words is too high.");
        }
    }

    void operator()() const {
        for (uindex_word_t i_word = 0; i_word < n_words; i_word++) {
            T word = in_pipe::read();
            if (low_pipe_active) {
                low_out_pipe::write(word);
            } else {
                high_out_pipe::write(word);
            }
        }
    }

  private:
    std::size_t n_words;
    bool low_pipe_active;
};

template <typename T, typename low_in_pipe, typename high_in_pipe, typename out_pipe,
          std::size_t max_n_words = std::numeric_limits<std::size_t>::max()>
class MergeSwitchKernel {
  public:
    using uindex_word_t = ac_int<std::bit_width(max_n_words), false>;

    MergeSwitchKernel(std::size_t n_words, bool low_pipe_active)
        : n_words(n_words), low_pipe_active(low_pipe_active) {
        if (n_words > max_n_words) {
            throw std::out_of_range("The number of requested pipe words is too high.");
        }
    }

    void operator()() const {
        for (uindex_word_t i_word = 0; i_word < n_words; i_word++) {
            T word;
            if (low_pipe_active) {
                word = low_in_pipe::read();
            } else {
                word = high_in_pipe::read();
            }
            out_pipe::write(word);
        }
    }

  private:
    std::size_t n_words;
    bool low_pipe_active;
};

} // namespace internal
} // namespace stencil
