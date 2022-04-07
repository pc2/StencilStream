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
    struct Cell {
        float ex, ey, hz, hz_sum;
        typename MaterialResolver::MaterialIdentifier mat_ident;
    };

    Kernel(Parameters const &parameters, MaterialResolver mat_resolver, Source source)
        : t_cutoff(parameters.t_cutoff()), t_detect(parameters.t_detect()), dt(parameters.dt()),
          mat_resolver(mat_resolver), source(source) {}

    static Cell halo() {
        Cell new_cell;
        new_cell.ex = 0;
        new_cell.ey = 0;
        new_cell.hz = 0;
        new_cell.hz_sum = 0;
        return new_cell;
    }

    Cell operator()(Stencil<Cell, stencil_radius> const &stencil) const {
        Cell cell = stencil[ID(0, 0)];

        CoefMaterial material = mat_resolver.identifier_to_material(cell.mat_ident);

        if ((stencil.stage & 0b1) == 0) {
            cell.ex *= material.ca;
            cell.ex += material.cb * (stencil[ID(0, 0)].hz - stencil[ID(0, -1)].hz);

            cell.ey *= material.ca;
            cell.ey += material.cb * (stencil[ID(-1, 0)].hz - stencil[ID(0, 0)].hz);
        } else {
            cell.hz *= material.da;
            cell.hz += material.db * (stencil[ID(0, 1)].ex - stencil[ID(0, 0)].ex +
                                      stencil[ID(0, 0)].ey - stencil[ID(1, 0)].ey);

            float current_time = (stencil.generation >> 1) * dt;

            if (stencil.id.c == stencil.grid_range.c / 2 &&
                stencil.id.r == stencil.grid_range.r / 2 && current_time < t_cutoff) {
                cell.hz += source.get_source_amplitude(stencil.stage, current_time);
            }

            if (current_time > t_detect) {
                cell.hz_sum += cell.hz * cell.hz;
            }
        }
        return cell;
    }

  private:
    float t_cutoff;
    float t_detect;
    float dt;
    MaterialResolver mat_resolver;
    Source source;
};