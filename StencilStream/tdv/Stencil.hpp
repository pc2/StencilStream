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
#include "../Stencil.hpp"
#include "Concepts.hpp"

namespace stencil {
namespace tdv {

template <typename Cell, uindex_t stencil_radius, TimeDependentValue TDV>
class Stencil : public stencil::Stencil<Cell, stencil_radius> {
  public:
    static constexpr uindex_t diameter = stencil::Stencil<Cell, stencil_radius>::diameter;

    Stencil(ID id, UID grid_range, uindex_t generation, uindex_t subgeneration,
            uindex_t i_processing_element, TDV tdv)
        : stencil::Stencil<Cell, stencil_radius>(id, grid_range, generation, subgeneration,
                                                 i_processing_element),
          tdv(tdv) {}

    Stencil(ID id, UID grid_range, uindex_t generation, uindex_t subgeneration,
            uindex_t i_processing_element, TDV tdv, Cell raw[diameter][diameter])
        : stencil::Stencil<Cell, stencil_radius>(id, grid_range, generation, subgeneration,
                                                 i_processing_element, raw),
          tdv(tdv) {}

    const TDV tdv;
};

} // namespace tdv
} // namespace stencil