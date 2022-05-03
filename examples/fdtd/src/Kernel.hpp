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

template <typename MaterialResolver, typename Source, typename TimeResolver> class Kernel {
  public:
    using MaterialCell = typename MaterialResolver::MaterialCell;

    Kernel(Parameters const &parameters, MaterialResolver mat_resolver, Source source,
           TimeResolver time_resolver)
        : t_cutoff(), t_detect(), dt(), source_radius_squared(), source_c(), source_r(),
          source_distance_bound(), double_center_cr(), mat_resolver(mat_resolver), source(source),
          time_resolver(time_resolver) {
        t_cutoff = parameters.t_cutoff();
        t_detect = parameters.t_detect();
        dt = parameters.dt();

        source_radius_squared = parameters.source_radius / parameters.dx;
        source_radius_squared *= source_radius_squared;

        source_c = parameters.source_c();
        source_r = parameters.source_r();
        source_distance_bound = parameters.source_radius / parameters.dx;
        source_distance_bound = source_distance_bound * source_distance_bound;
        source_distance_bound -= source_c * source_c + source_r * source_r;

        double_center_cr = parameters.grid_range()[0];
    }

    MaterialCell operator()(Stencil<MaterialCell, 1> const &stencil) const {
        MaterialCell cell = stencil[ID(0, 0)];

        index_t c = stencil.id.c;
        index_t r = stencil.id.r;
        index_t center_distance_score = c * (c - double_center_cr) + r * (r - double_center_cr);
        index_t source_distance_score = c * (c - 2 * source_c) + r * (r - 2 * source_r);

        CoefMaterial material =
            mat_resolver.get_material_coefficients(stencil, center_distance_score);

        if ((stencil.stage & 0b1) == 0) {
            cell.cell.ex *= material.ca;
            cell.cell.ex += material.cb * (stencil[ID(0, 0)].cell.hz - stencil[ID(0, -1)].cell.hz);

            cell.cell.ey *= material.ca;
            cell.cell.ey += material.cb * (stencil[ID(-1, 0)].cell.hz - stencil[ID(0, 0)].cell.hz);
        } else {
            cell.cell.hz *= material.da;
            cell.cell.hz += material.db * (stencil[ID(0, 1)].cell.ex - stencil[ID(0, 0)].cell.ex +
                                           stencil[ID(0, 0)].cell.ey - stencil[ID(1, 0)].cell.ey);

            float current_time = time_resolver.template resolve_time<MaterialCell>(stencil);

            if (source_distance_score <= source_distance_bound && current_time < t_cutoff) {
                float interp_factor;
                if (source_radius_squared != 0) {
                    index_t cell_distance_squared =
                        source_distance_score + source_c * source_c + source_r * source_r;
                    // cell_distance_squared == (distance / dx)^2
                    interp_factor = 1.0 - float(cell_distance_squared) / source_radius_squared;
                } else {
                    interp_factor = 1.0;
                }

                cell.cell.hz +=
                    interp_factor * source.get_source_amplitude(stencil.stage, current_time);
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

    float source_radius_squared;
    index_t source_c, source_r, source_distance_bound;
    index_t double_center_cr;

    MaterialResolver mat_resolver;
    Source source;
    TimeResolver time_resolver;
};