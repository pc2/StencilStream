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
#include "../Parameters.hpp"
#include "../defines.hpp"
#include "SourceFunction.hpp"

class TimeLUTSource {
  public:
    TimeLUTSource(Parameters const &parameters, uindex_t i_generation)
        : starting_generation(i_generation), time(), tau(parameters.tau), omega(parameters.omega()),
          t_0(parameters.t_0()) {
        for (uindex_t i = 0; i < gens_per_pass; i++) {
            time[i] = (i_generation + i) * parameters.dt();
        }
    }

    template <typename Cell> float get_source_amplitude(Stencil<Cell, 1> const &stencil) const {
        float current_time = time[stencil.i_processing_element >> 1];
        return calc_source_amplitude(current_time, t_0, tau, omega);
    }

  private:
    uindex_t starting_generation;
    float time[gens_per_pass];
    float tau;
    float omega;
    float t_0;
};
