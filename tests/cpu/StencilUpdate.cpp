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
#include "../StencilUpdateTest.hpp"
#include "../constants.hpp"
#include <StencilStream/cpu/StencilUpdate.hpp>

using namespace sycl;
using namespace stencil;
using namespace stencil::cpu;

using StencilUpdateImpl = StencilUpdate<FPGATransFunc<1>, tdv::InlineSupplier<GenerationFunction>>;
using GridImpl = Grid<Cell>;

static_assert(concepts::StencilUpdate<StencilUpdateImpl, FPGATransFunc<1>,
                                      tdv::InlineSupplier<GenerationFunction>, GridImpl>);

TEST_CASE("cpu::StencilUpdate", "[cpu::StencilUpdate]") {
    test_stencil_update<GridImpl, StencilUpdateImpl>(64, 64, 0, 1);
    test_stencil_update<GridImpl, StencilUpdateImpl>(64, 64, 0, 1);
    test_stencil_update<GridImpl, StencilUpdateImpl>(64, 64, 32, 64);
}