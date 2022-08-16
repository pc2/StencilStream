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
#include "../Concepts.hpp"

namespace stencil {
namespace tdv {

template <ValueFunction F> class InlineSupplier {
  public:
    struct GlobalState {
        using LocalState = GlobalState;
        using Value = typename F::Value;

        GlobalState build_local_state() const { return *this; }

        Value get_value(uindex_t i) const { return function(i_generation + i); }

        F function;
        uindex_t i_generation;
    };

    InlineSupplier(F function) : function(function) {}

    void prepare_range(uindex_t i_generation, uindex_t n_generations) {}

    GlobalState build_global_state(cl::sycl::handler &cgh, uindex_t i_generation,
                                   uindex_t n_generations) {
        return GlobalState{.function = function, .i_generation = i_generation};
    }

  private:
    F function;
};

} // namespace tdv
} // namespace stencil