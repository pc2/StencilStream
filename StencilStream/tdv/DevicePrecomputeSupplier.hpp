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
#include "../Concepts.hpp"
#include <array>
#include <cassert>
#include <numeric>

namespace stencil {
namespace tdv {

template <concepts::tdv::ValueFunction F, uindex_t max_n_generations>
class DevicePrecomputeSupplier {
  public:
    DevicePrecomputeSupplier(F function) : function(function) {}

    using Value = typename F::Value;

    struct KernelArgument {
        struct LocalState {
            using Value = Value;

            Value get_value(uindex_t i) const { return values[i]; }

            Value values[max_n_generations];
        };

        LocalState build_local_state() const {
            LocalState state;
            for (uindex_t i = 0; i < max_n_generations; i++) {
                state.values[i] = function(generation_offset + i);
            }
            return state;
        }

        uindex_t generation_offset;
        F function;
    };

    void prepare_range(uindex_t i_generation, uindex_t n_generations) {}

    KernelArgument build_kernel_argument(sycl::handler &cgh, uindex_t i_generation,
                                         uindex_t n_generations) {

        return KernelArgument{.generation_offset = i_generation, .function = function};
    }

  private:
    F function;
};

} // namespace tdv
} // namespace stencil