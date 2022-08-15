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
#include <array>
#include <cassert>

namespace stencil {
namespace tdv {

template <typename V, uindex_t max_n_generations> class ValueBuffer {
  public:
    ValueBuffer(std::array<V, max_n_generations> values) : values(values) {}

    using LocalState = ValueBuffer<V, max_n_generations>;

    ValueBuffer prepare_local_state() const { return *this; }

    V get_value(uindex_t i) const { return values[i]; }

  private:
    std::array<V, max_n_generations> values;
};

template <ValueFunction F, uindex_t max_n_generations> class OfflineValueSupplier {
  public:
    OfflineValueSupplier(F function) : function(function) {}

    using Value = typename F::Value;

    using GlobalState = ValueBuffer<Value, max_n_generations>;

    GlobalState prepare_global_state(uindex_t i_generation, uindex_t n_generations) const {
        assert(n_generations <= max_n_generations);

        std::array<Value, max_n_generations> values;
        for (uindex_t i = 0; i < n_generations; i++) {
            values[i] = function(i + i_generation);
        }

        return GlobalState(values);
    }

  private:
    F function;
};

} // namespace tdv
} // namespace stencil