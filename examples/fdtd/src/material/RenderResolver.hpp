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
#include "../Cell.hpp"
#include "../Parameters.hpp"
#include "Material.hpp"

class RenderResolver {
  public:
    struct MaterialCell {
        Cell cell;

        static MaterialCell halo() { return MaterialCell{Cell::halo()}; }

        static MaterialCell from_parameters(Parameters const &parameters, uindex_t material_index) {
            // No computations needed here, since no material information is stored in the cells.
            return MaterialCell{Cell::halo()};
        }
    };

    /**
     * Derivation of the distance measuring system:
     *
     * The goal is to reduce the number of operations executed on the FPGA
     */

    RenderResolver(Parameters const &parameters) : distance_bound(0.0), materials() {
        float r = parameters.disk_radius;
        float dx = parameters.dx;
        float w = parameters.grid_range()[0];
        distance_bound = 2 * (r / dx) * (r / dx) - (w * w);

        materials[0] = CoefMaterial::from_relative_material(RelMaterial::perfect_metal(),
                                                            parameters.dx, parameters.dt());
        materials[1] = CoefMaterial::from_relative_material(
            RelMaterial{parameters.mu_r, parameters.eps_r, parameters.sigma}, parameters.dx,
            parameters.dt());
    }

    CoefMaterial get_material_coefficients(Stencil<MaterialCell, 1> const &stencil) const {
        uindex_t c = stencil.id.c;
        uindex_t r = stencil.id.r;
        uindex_t w = stencil.grid_range.c;
        uindex_t distance_score = 2 * (c * (c - w) + r * (r - w));

        if (distance_score <= distance_bound) {
            return materials[1];
        } else {
            return materials[0];
        }
    }

  private:
    uindex_t distance_bound;
    CoefMaterial materials[2];
};
