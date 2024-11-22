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
#include "../Cell.hpp"
#include "../Parameters.hpp"
#include "Material.hpp"

class RenderResolver {
  public:
    struct MaterialCell {
        Cell cell;

        static MaterialCell halo() { return MaterialCell{Cell::halo()}; }

        static MaterialCell from_parameters(Parameters const &parameters, size_t material_index) {
            // No computations needed here, since no material information is stored in the cells.
            return MaterialCell{Cell::halo()};
        }
    };

    RenderResolver(Parameters const &parameters) : distance_bounds(), materials() {
        float dx = parameters.dx;
        float center_r = parameters.grid_range()[0] / 2;

        float radius = 0.0;
        for (size_t i = 0; i < max_n_rings + 1; i++) {
            if (i < parameters.rings.size()) {
                Parameters::RingParameter const &ring = parameters.rings[i];
                radius += ring.radius;

                distance_bounds[i] = (radius / dx) * (radius / dx) - 2 * center_r * center_r;
                materials[i] = CoefMaterial::from_relative_material(ring.material, parameters.dx,
                                                                    parameters.dt());
            } else {
                distance_bounds[i] = std::numeric_limits<float>::infinity();
                materials[i] = CoefMaterial::perfect_metal();
            }
        }
    }

    CoefMaterial get_material_coefficients(Stencil<MaterialCell, 1, float> const &stencil,
                                           float distance_score) const {
#pragma unroll
        for (uindex_ring_t i = 0; i < max_n_rings + 1; i++) {
            if (distance_score <= distance_bounds[i]) {
                return materials[i];
            }
        }
        return materials[max_n_rings];
    }

  private:
    float distance_bounds[max_n_rings + 1];
    CoefMaterial materials[max_n_rings + 1];
};
