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
#include <StencilStream/Stencil.hpp>

class FDTDKernel {
    float disk_radius;
    float tau;
    float omega;
    float t_0;
    float t_cutoff;
    float t_detect;
    float dx;
    float dt;
    Material vacuum;

  public:
    FDTDKernel(Parameters const &parameters)
        : disk_radius(parameters.disk_radius), tau(parameters.tau), omega(parameters.omega()),
          t_0(parameters.t_0()), t_cutoff(parameters.t_cutoff()), t_detect(parameters.t_detect()),
          dx(parameters.dx), dt(parameters.dt()), vacuum(parameters.vacuum()) {}

    static FDTDCell halo() {
        FDTDCell new_cell;
        new_cell.ex = 0;
        new_cell.ey = 0;
        new_cell.hz = 0;
        new_cell.hz_sum = 0;
        new_cell.distance = INFINITY;
        return new_cell;
    }

    FDTDCell operator()(Stencil<FDTDCell, stencil_radius> const &stencil) const {
        FDTDCell cell = stencil[ID(0, 0)];

        if (cell.distance < disk_radius) {
            if ((stencil.stage & 0b1) == 0) {
                cell.ex *= vacuum.ca;
                cell.ex += vacuum.cb * (stencil[ID(0, 0)].hz - stencil[ID(0, -1)].hz);

                cell.ey *= vacuum.ca;
                cell.ey += vacuum.cb * (stencil[ID(-1, 0)].hz - stencil[ID(0, 0)].hz);
            } else {
                cell.hz *= vacuum.da;
                cell.hz += vacuum.db * (stencil[ID(0, 1)].ex - stencil[ID(0, 0)].ex +
                                        stencil[ID(0, 0)].ey - stencil[ID(1, 0)].ey);

                float current_time = (stencil.generation >> 1) * dt;
                if (cell.distance < dx && current_time < t_cutoff) {
                    float wave_progress = (current_time - t_0) / tau;
                    cell.hz += cl::sycl::cos(omega * current_time) *
                               cl::sycl::exp(-1 * wave_progress * wave_progress);
                }

                if (current_time > t_detect) {
                    cell.hz_sum += cell.hz * cell.hz;
                }
            }
        }
        return cell;
    }
};