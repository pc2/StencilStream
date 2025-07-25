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
#include <functional>
#include <mpi.h>
#include <numeric>
#include <pc2/queue_extensions.hpp>
#include <stdint.h>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#if defined(STENCILSTREAM_NAMED_KERNELS)
    #define STENCILSTREAM_NAMED_SINGLE_TASK(Name, argument) single_task<class Name>(argument)
#else
    #define STENCILSTREAM_NAMED_SINGLE_TASK(Name, argument) single_task(argument)
#endif

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

inline std::vector<sycl::queue>
alloc_queues(std::function<int(const sycl::device &)> device_selector, std::size_t n_queues) {
    std::vector<sycl::queue> queues;
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        queues = sycl::ext::pc2::mpi_queues<std::function<int(const sycl::device &)>,
                                            sycl::property_list>(
            device_selector, n_queues, {sycl::property::queue::in_order{}});
    } else {
        sycl::device device(device_selector);
        for (std::size_t i_queue = 0; i_queue < n_queues; i_queue++) {
            queues.push_back(sycl::queue(device, {sycl::property::queue::in_order{}}));
        }
    }
    return queues;
}

} // namespace internal
} // namespace stencil