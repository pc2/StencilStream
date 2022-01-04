/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "AbstractExecutor.hpp"
#include "RuntimeSample.hpp"
#include <optional>

namespace stencil {

template <typename T, uindex_t stencil_radius, typename TransFunc>
class RuntimeSampleExecutor : public AbstractExecutor<T, stencil_radius, TransFunc> {
public:
    using Parent = AbstractExecutor<T, stencil_radius, TransFunc>;

    RuntimeSampleExecutor(T halo_value, TransFunc trans_func) : Parent(halo_value, trans_func), runtime_sample(), analysis_enabled(true) {}

    /**
     * \brief Return a reference to the runtime information struct.
     *
     * \return The collected runtime information.
     */
    RuntimeSample &get_runtime_sample() { return runtime_sample; }
private:
    RuntimeSample runtime_sample;
    bool analysis_enabled;
};

}