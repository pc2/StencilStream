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

using namespace stencil;

template <typename MaterialResolver> class Kernel {
  public:
    using Cell = typename MaterialResolver::MaterialCell;
    using TimeDependentValue = float;

    static constexpr size_t stencil_radius = 1;
    static constexpr size_t n_subiterations = 2;

    Kernel(Parameters const &parameters, MaterialResolver mat_resolver)
        : dt(parameters.dt()), t_0(parameters.t_0()), tau(parameters.tau),
          omega(parameters.omega()), cutoff_iteration(), detect_iteration(),
          source_radius_squared(), source_c(), source_r(), source_distance_bound(),
          double_center_rc(), mat_resolver(mat_resolver) {
        cutoff_iteration = std::floor(parameters.t_cutoff() / parameters.dt());
        detect_iteration = std::floor(parameters.t_detect() / parameters.dt());

        source_radius_squared = parameters.source_radius / parameters.dx;
        source_radius_squared *= source_radius_squared;

        source_r = parameters.source_r();
        source_c = parameters.source_c();
        source_distance_bound = parameters.source_radius / parameters.dx;
        source_distance_bound = source_distance_bound * source_distance_bound;
        source_distance_bound -= source_c * source_c + source_r * source_r;

        double_center_rc = parameters.grid_range()[0];
    }

    float get_time_dependent_value(size_t i_iteration) const {
        float current_time = i_iteration * dt;
        float wave_progress = (current_time - t_0) / tau;
        return sycl::cos(omega * current_time) * sycl::exp(-1 * wave_progress * wave_progress);
    }

#if defined(STENCILSTREAM_BACKEND_CUDA_SoA)
    Cell operator()(Stencil<Cell, 1, float> const &stencil) const {
        Cell cell = stencil[0][0];

        float r = stencil.id[0];
        float c = stencil.id[1];
        float center_distance_score = r * (r - double_center_rc) + c * (c - double_center_rc);
        float source_distance_score = r * (r - 2 * source_r) + c * (c - 2 * source_c);

        CoefMaterial material =
            mat_resolver.get_material_coefficients(stencil, center_distance_score);
        if (stencil.subiteration == 0) {
            cell.ex *= material.ca;
            cell.ex += material.cb * (stencil[0][0].hz - stencil[0][-1].hz);

            cell.ey *= material.ca;
            cell.ey += material.cb * (stencil[-1][0].hz - stencil[0][0].hz);
        } else {
            cell.hz *= material.da;
            cell.hz += material.db *
                       (stencil[0][1].ex - stencil[0][0].ex + stencil[0][0].ey - stencil[1][0].ey);

            if (source_distance_score <= source_distance_bound &&
                stencil.iteration <= cutoff_iteration) {
                float interp_factor;
                if (source_radius_squared != 0) {
                    float cell_distance_squared =
                        source_distance_score + source_c * source_c + source_r * source_r;
                    // cell_distance_squared == (distance / dx)^2
                    interp_factor = 1.0 - float(cell_distance_squared) / source_radius_squared;
                } else {
                    interp_factor = 1.0;
                }

                float source_amplitude = stencil.time_dependent_value;
                cell.hz += interp_factor * source_amplitude;
            }

            if (stencil.iteration > detect_iteration) {
                cell.hz_sum += cell.hz * cell.hz;
            }
        }
        return cell;
    }
#else
    Cell operator()(Stencil<Cell, 1, float> const &stencil) const {
        Cell cell = stencil[0][0];

        float r = stencil.id[0];
        float c = stencil.id[1];
        float center_distance_score = r * (r - double_center_rc) + c * (c - double_center_rc);
        float source_distance_score = r * (r - 2 * source_r) + c * (c - 2 * source_c);

        CoefMaterial material =
            mat_resolver.get_material_coefficients(stencil, center_distance_score);

        if (stencil.subiteration == 0) {
            cell.cell.ex *= material.ca;
            cell.cell.ex += material.cb * (stencil[0][0].cell.hz - stencil[0][-1].cell.hz);

            cell.cell.ey *= material.ca;
            cell.cell.ey += material.cb * (stencil[-1][0].cell.hz - stencil[0][0].cell.hz);
        } else {
            cell.cell.hz *= material.da;
            cell.cell.hz += material.db * (stencil[0][1].cell.ex - stencil[0][0].cell.ex +
                                           stencil[0][0].cell.ey - stencil[1][0].cell.ey);

            if (source_distance_score <= source_distance_bound &&
                stencil.iteration <= cutoff_iteration) {
                float interp_factor;
                if (source_radius_squared != 0) {
                    float cell_distance_squared =
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
#endif

  private:
    float dt, t_0, tau, omega;

    size_t cutoff_iteration;
    size_t detect_iteration;

    float source_radius_squared;
    float source_r, source_c, source_distance_bound;
    float double_center_rc;

    MaterialResolver mat_resolver;
};