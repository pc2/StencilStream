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
#include "defines.hpp"
#include "Material.hpp"
#include <StencilStream/Stencil.hpp>
#include "Parameters.hpp"

class LUTKernel {
    float disk_radius;
    float tau;
    float omega;
    float t_0;
    float t_cutoff;
    float t_detect;
    float dx;
    float dt;
    CoefMaterial materials[max_materials];

  public:

    struct Cell {
        float ex, ey, hz, hz_sum;
        uint32_t material_index;
    };

    LUTKernel(Parameters const &parameters)
        : disk_radius(parameters.disk_radius), tau(parameters.tau), omega(parameters.omega()),
          t_0(parameters.t_0()), t_cutoff(parameters.t_cutoff()), t_detect(parameters.t_detect()),
          dx(parameters.dx), dt(parameters.dt()), materials{} {
        materials[0] = CoefMaterial::from_relative(RelMaterial { std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), 0.0}, dx, dt);
        materials[1] = CoefMaterial::from_relative(RelMaterial { 1.0, 1.0, 0.0}, dx, dt);
    }

    static Cell halo() {
        Cell new_cell;
        new_cell.ex = 0;
        new_cell.ey = 0;
        new_cell.hz = 0;
        new_cell.hz_sum = 0;
        new_cell.material_index = 0;
        return new_cell;
    }

    Cell operator()(Stencil<Cell, stencil_radius> const &stencil) const {
        Cell cell = stencil[ID(0, 0)];

        uint32_t material_index = cell.material_index;
        if (material_index >= max_materials) {
            material_index = 0;
        }

        if ((stencil.stage & 0b1) == 0) {
            cell.ex *= materials[material_index].ca;
            cell.ex += materials[material_index].cb * (
                stencil[ID(0, 0)].hz - stencil[ID(0, -1)].hz
            );

            cell.ey *= materials[material_index].ca;
            cell.ey += materials[material_index].cb * (
                stencil[ID(-1, 0)].hz - stencil[ID(0, 0)].hz
            );
        } else {
            cell.hz *= materials[material_index].da;
            cell.hz += materials[material_index].db * (
                stencil[ID(0, 1)].ex - stencil[ID(0, 0)].ex +
                stencil[ID(0, 0)].ey - stencil[ID(1, 0)].ey
            );

            float current_time = (stencil.generation >> 1) * dt;
            if (stencil.id.c == stencil.grid_range.c / 2 && stencil.id.r == stencil.grid_range.r / 2 && current_time < t_cutoff) {
                float wave_progress = (current_time - t_0) / tau;
                cell.hz += cl::sycl::cos(omega * current_time) *
                           cl::sycl::exp(-1 * wave_progress * wave_progress);
            }

            if (current_time > t_detect) {
                cell.hz_sum += cell.hz * cell.hz;
            }
        }
        return cell;
    }
};