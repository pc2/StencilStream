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
#include <array>
#include <numeric>
#include <stdint.h>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

namespace stencil {
namespace internal {

/**
 * \brief A container with padding to the next power of two.
 *
 * Wrapping a type in this template ensures that the resulting size is a power of two.
 *
 * \tparam T The contained type.
 */
template <typename T> struct Padded {
    T value;
} __attribute__((aligned(std::bit_ceil(sizeof(T)))));

template <typename T> inline constexpr T int_ceil_div(T a, T b) {
    return a / b + ((a % b == 0) ? 0 : 1);
}

struct kernel_input_ch0 {
    static constexpr unsigned id = 0;
};

struct kernel_output_ch0 {
    static constexpr unsigned id = 1;
};

struct kernel_input_ch1 {
    static constexpr unsigned id = 2;
};

struct kernel_output_ch1 {
    static constexpr unsigned id = 3;
};

struct kernel_input_ch2 {
    static constexpr unsigned id = 4;
};

struct kernel_output_ch2 {
    static constexpr unsigned id = 5;
};

struct kernel_input_ch3 {
    static constexpr unsigned id = 6;
};

struct kernel_output_ch3 {
    static constexpr unsigned id = 7;
};

constexpr std::size_t pipeword_size = 32;
using pipeword_t = std::array<uint8_t, pipeword_size> __attribute__((aligned(pipeword_size)));

} // namespace internal
} // namespace stencil