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
#include "Index.hpp"
#include <CL/sycl.hpp>

namespace stencil {
template <typename Cell, typename Accessor, sycl::access::mode access_mode,
          uindex_t current_subdim = 0>
class AccessorSubscript {
  public:
    static constexpr uindex_t dimensions = Accessor::dimensions;
    AccessorSubscript(Accessor &ac, uindex_t i) : ac(ac), id_prefix() {
        id_prefix[current_subdim] = i;
    }

    AccessorSubscript(Accessor &ac, sycl::id<dimensions> id_prefix, uindex_t i)
        : ac(ac), id_prefix(id_prefix) {
        id_prefix[current_subdim] = i;
    }

    AccessorSubscript<Cell, Accessor, access_mode, current_subdim + 1> operator[](uindex_t i)
        requires(current_subdim < dimensions - 2)
    {
        return AccessorSubscript(ac, id_prefix, i);
    }

    Cell const &operator[](uindex_t i)
        requires(current_subdim == dimensions - 2 && access_mode == sycl::access::mode::read)
    {
        sycl::id<dimensions> id = id_prefix;
        id[current_subdim + 1] = i;
        return ac[id];
    }

    Cell &operator[](uindex_t i)
        requires(current_subdim == dimensions - 2 && access_mode != sycl::access::mode::read)
    {
        sycl::id<dimensions> id = id_prefix;
        id[current_subdim + 1] = i;
        return ac[id];
    }

  private:
    Accessor &ac;
    sycl::id<dimensions> id_prefix;
};
} // namespace stencil