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
#include "Parameters.hpp"
#include "defines.hpp"
#include "material/Material.hpp"
#include <StencilStream/Stencil.hpp>

template <typename MaterialResolver, typename Source> class Kernel {
  public:
    using MaterialCell = typename MaterialResolver::MaterialCell;

    Kernel(Parameters const &parameters, MaterialResolver mat_resolver, Source source)
        : t_cutoff(parameters.t_cutoff()), t_detect(parameters.t_detect()), dt(parameters.dt()),
          source_c(parameters.source_c()), source_r(parameters.source_r()),
          mat_resolver(mat_resolver), source(source) {}

    MaterialCell operator()(Stencil<MaterialCell, 1> const &stencil) const {
        MaterialCell cell = stencil[ID(0, 0)];

        CoefMaterial material = mat_resolver.get_material_coefficients(stencil);

        if ((stencil.stage & 0b1) == 0) {
            cell.cell.ex *= material.ca;
            cell.cell.ex += material.cb * (stencil[ID(0, 0)].cell.hz - stencil[ID(0, -1)].cell.hz);

            cell.cell.ey *= material.ca;
            cell.cell.ey += material.cb * (stencil[ID(-1, 0)].cell.hz - stencil[ID(0, 0)].cell.hz);
        } else {
            cell.cell.hz *= material.da;
            cell.cell.hz += material.db * (stencil[ID(0, 1)].cell.ex - stencil[ID(0, 0)].cell.ex +
                                           stencil[ID(0, 0)].cell.ey - stencil[ID(1, 0)].cell.ey);

            float current_time = (stencil.generation >> 1) * dt;

            if (stencil.id.c == source_c &&
                stencil.id.r == source_r && current_time < t_cutoff) {
                cell.cell.hz += source.get_source_amplitude(stencil.stage, current_time);
            }

            if (current_time > t_detect) {
                cell.cell.hz_sum += cell.cell.hz * cell.cell.hz;
            }
        }
        return cell;
    }

  private:
    float t_cutoff;
    float t_detect;
    float dt;
    uindex_t source_c, source_r;
    MaterialResolver mat_resolver;
    Source source;
};