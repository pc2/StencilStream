/*
 * Copyright © 2020-2023 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "../Concepts.hpp"
#include <array>
#include <cassert>
#include <numeric>

namespace stencil {
namespace tdv {

template <ValueFunction F, uindex_t max_n_generations> class HostPrecomputeSupplier {
  public:
    HostPrecomputeSupplier(F function)
        : function(function), generation_offset(0), value_buffer(cl::sycl::range<1>(1)) {
        value_buffer.template get_access<cl::sycl::access::mode::discard_write>()[0] = function(0);
    }

    using Value = typename F::Value;

    struct KernelArgument {
        struct LocalState {
            using Value = Value;

            Value get_value(uindex_t i) const { return values[i]; }

            Value values[max_n_generations];
        };

        LocalState build_local_state() const {
            uindex_t n_values =
                std::min(uindex_t(max_n_generations), uindex_t(ac.get_range()[0] - buffer_offset));

            LocalState state;
            for (uindex_t i = 0; i < n_values; i++) {
                state.values[i] = ac[buffer_offset + i];
            }
            return state;
        }

        uindex_t buffer_offset;
        cl::sycl::accessor<Value, 1, cl::sycl::access::mode::read> ac;
    };

    void prepare_range(uindex_t i_generation, uindex_t n_generations) {
        generation_offset = i_generation;
        value_buffer = cl::sycl::range<1>(n_generations);

        auto ac = value_buffer.template get_access<cl::sycl::access::mode::discard_write>();
        for (uindex_t i = 0; i < n_generations; i++) {
            ac[i] = function(i_generation + i);
        }
    }

    KernelArgument build_kernel_argument(cl::sycl::handler &cgh, uindex_t i_generation,
                                   uindex_t n_generations) {
        assert(n_generations <= max_n_generations);
        assert(i_generation >= generation_offset);
        assert(i_generation + n_generations <= generation_offset + value_buffer.get_range()[0]);

        return KernelArgument{.buffer_offset = i_generation - generation_offset,
                           .ac =
                               value_buffer.template get_access<cl::sycl::access::mode::read>(cgh)};
    }

  private:
    F function;
    uindex_t generation_offset;
    cl::sycl::buffer<Value, 1> value_buffer;
};

} // namespace tdv
} // namespace stencil