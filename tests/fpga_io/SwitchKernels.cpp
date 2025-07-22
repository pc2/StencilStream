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
#include <StencilStream/fpga_io/SwitchKernels.hpp>
#include <catch2/catch_all.hpp>

constexpr std::size_t max_n_words = 255;

void test_fork_switch_kernel(std::size_t n_words, bool low_pipe_active) {
    using in_pipe = sycl::pipe<class fork_switch_in_pipe_id, std::size_t>;
    using low_out_pipe = sycl::pipe<class fork_switch_low_out_pipe_id, std::size_t>;
    using high_out_pipe = sycl::pipe<class fork_switch_high_out_pipe_id, std::size_t>;
    using Kernel = stencil::fpga_io::ForkSwitchKernel<std::size_t, in_pipe, low_out_pipe, high_out_pipe, max_n_words>;

    sycl::queue queue;
    queue.single_task([=]() {
        for (std::size_t i_word = 0; i_word < n_words; i_word++) {
            in_pipe::write(i_word);
        }
    });

    queue.single_task(Kernel(n_words, low_pipe_active));

    sycl::buffer<std::size_t, 1> out_buffer = sycl::range<1>(n_words);
    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor ac(out_buffer, cgh, sycl::write_only);

        cgh.single_task([=]() {
            for (std::size_t i_word = 0; i_word < n_words; i_word++) {
                ac[i_word] = (low_pipe_active) ? low_out_pipe::read() : high_out_pipe::read();
            }
        });
    });

    sycl::host_accessor ac(out_buffer, sycl::read_only);
    for (std::size_t i_word = 0; i_word < n_words; i_word++) {
        REQUIRE(ac[i_word] == i_word);
    }
}

TEST_CASE("fpga_io::ForkSwitchKernel", "[fpga_io]") {
    test_fork_switch_kernel(2, false);
    test_fork_switch_kernel(max_n_words, false);
    test_fork_switch_kernel(2, true);
    test_fork_switch_kernel(max_n_words, true);
}

void test_merge_switch_kernel(std::size_t n_words, bool low_pipe_active) {
    using low_in_pipe = sycl::pipe<class merge_switch_low_in_pipe_id, std::size_t>;
    using high_in_pipe = sycl::pipe<class merge_switch_high_in_pipe_id, std::size_t>;
    using out_pipe = sycl::pipe<class merge_switch_out_pipe_id, std::size_t>;
    using Kernel = stencil::fpga_io::MergeSwitchKernel<std::size_t, low_in_pipe, high_in_pipe, out_pipe, max_n_words>;

    sycl::queue queue;
    queue.single_task([=]() {
        for (std::size_t i_word = 0; i_word < n_words; i_word++) {
            if (low_pipe_active) {
                low_in_pipe::write(i_word);
            } else {
                high_in_pipe::write(i_word);
            }
        }
    });

    queue.single_task(Kernel(n_words, low_pipe_active));

    sycl::buffer<std::size_t, 1> out_buffer = sycl::range<1>(n_words);
    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor ac(out_buffer, cgh, sycl::write_only);

        cgh.single_task([=]() {
            for (std::size_t i_word = 0; i_word < n_words; i_word++) {
                ac[i_word] = out_pipe::read();
            }
        });
    });

    sycl::host_accessor ac(out_buffer, sycl::read_only);
    for (std::size_t i_word = 0; i_word < n_words; i_word++) {
        REQUIRE(ac[i_word] == i_word);
    }
}

TEST_CASE("fpga_io::MergeSwitchKernel", "[fpga_io]") {
    test_merge_switch_kernel(2, false);
    test_merge_switch_kernel(max_n_words, false);
    test_merge_switch_kernel(2, true);
    test_merge_switch_kernel(max_n_words, true);
}