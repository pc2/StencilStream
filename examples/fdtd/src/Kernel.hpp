/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include <StencilStream/BaseTransitionFunction.hpp>
#include <StencilStream/Stencil.hpp>

template <typename MaterialResolver> class Kernel {
  public:
    using Cell = typename MaterialResolver::MaterialCell;
    using TimeDependentValue = float;

    static constexpr uindex_t stencil_radius = 1;
    static constexpr uindex_t n_subiterations = 2;

    Kernel(Parameters const &parameters, MaterialResolver mat_resolver)
        : dt(parameters.dt()), t_0(parameters.t_0()), tau(parameters.tau),
          omega(parameters.omega()), cutoff_iteration(), detect_iteration(),
          source_radius_squared(), source_c(), source_r(), source_distance_bound(),
          double_center_cr(), mat_resolver(mat_resolver) {
        cutoff_iteration = std::floor(parameters.t_cutoff() / parameters.dt());
        detect_iteration = std::floor(parameters.t_detect() / parameters.dt());

        source_radius_squared = parameters.source_radius / parameters.dx;
        source_radius_squared *= source_radius_squared;

        source_c = parameters.source_c();
        source_r = parameters.source_r();
        source_distance_bound = parameters.source_radius / parameters.dx;
        source_distance_bound = source_distance_bound * source_distance_bound;
        source_distance_bound -= source_c * source_c + source_r * source_r;

        double_center_cr = parameters.grid_range()[0];
    }

    float get_time_dependent_value(uindex_t i_iteration) const {
        float current_time = i_iteration * dt;
        float wave_progress = (current_time - t_0) / tau;
        return cl::sycl::cos(omega * current_time) *
               cl::sycl::exp(-1 * wave_progress * wave_progress);
    }

    Cell operator()(Stencil<Cell, 1, float> const &stencil) const {
        Cell cell = stencil[ID(0, 0)];

        index_t c = stencil.id.c;
        index_t r = stencil.id.r;
        index_t center_distance_score = c * (c - double_center_cr) + r * (r - double_center_cr);
        index_t source_distance_score = c * (c - 2 * source_c) + r * (r - 2 * source_r);

        CoefMaterial material =
            mat_resolver.get_material_coefficients(stencil, center_distance_score);

        if (stencil.subiteration == 0) {
            cell.cell.ex *= material.ca;
            cell.cell.ex += material.cb * (stencil[ID(0, 0)].cell.hz - stencil[ID(0, -1)].cell.hz);

            cell.cell.ey *= material.ca;
            cell.cell.ey += material.cb * (stencil[ID(-1, 0)].cell.hz - stencil[ID(0, 0)].cell.hz);
        } else {
            cell.cell.hz *= material.da;
            cell.cell.hz += material.db * (stencil[ID(0, 1)].cell.ex - stencil[ID(0, 0)].cell.ex +
                                           stencil[ID(0, 0)].cell.ey - stencil[ID(1, 0)].cell.ey);

            if (source_distance_score <= source_distance_bound &&
                stencil.iteration <= cutoff_iteration) {
                float interp_factor;
                if (source_radius_squared != 0) {
                    index_t cell_distance_squared =
                        source_distance_score + source_c * source_c + source_r * source_r;
                    // cell_distance_squared == (distance / dx)^2
                    interp_factor = 1.0 - float(cell_distance_squared) / source_radius_squared;
                } else {
                    interp_factor = 1.0;
                }

                float source_amplitude = stencil.time_dependent_value;
                cell.cell.hz += interp_factor * source_amplitude;
            }

            if (stencil.iteration > detect_iteration) {
                cell.cell.hz_sum += cell.cell.hz * cell.cell.hz;
            }
        }
        return cell;
    }

  private:
    float dt, t_0, tau, omega;

    uindex_t cutoff_iteration;
    uindex_t detect_iteration;

    float source_radius_squared;
    index_t source_c, source_r, source_distance_bound;
    index_t double_center_cr;

    MaterialResolver mat_resolver;
};