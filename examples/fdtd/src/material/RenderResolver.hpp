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
     * A cell has the parametrized material iff its distance to the center of the grid is within a
     * certain radius. Let c be the column index, r be the row index, w be the width or height of
     * the grid, r be the radius of the disk in meters and dx be the width or height of a cell in
     * meters. Then, we have:
     *
     *     Cell is in the disk <=> dx * sqrt((c - w/2)^2 + (r - w/2)^2) <= r
     *
     * The goal now is to to reduce the number of computations that have to be repeated and
     * therefore be executed on the FPGA. The values handled by the FPGA should also be integers
     * since the column and row indices are given as integers and integer-to-float conversions are
     * expensive. We have:
     *
     *     Cell is in the disk
     *     <=> dx * sqrt((c - w/2)^2 + (r - w/2)^2) <= r
     *     <=> (c - w/2)^2 + (r - w/2)^2 <= (r/dx)^2
     *     <=> (2c - w)^2 + (2r - w)^2 <= 4*(r/dx)^2
     *     <=> (4c^2 - 4wc + w^2) + (4r^2 - 4wr + w^2) <= 4*(r/dx)^2
     *     <=> 4*(c^2 - wc + r^2 - wr) <= 4*(r/dx)^2 - 2w^2
     *     <=> 2*(c * (c - w) + r * (r - w)) <= 2*(r/dx)^2 - w^2
     *
     * All intermediate values on the left-hand side are integers and the right-hand side expression
     * can be rounded to the nearest integer to form an approximation. When we therefore precompute
     * the right-hand side, we have reduced the bound check to six integer operations and one
     * comparison to a runtime constant.
     */

    RenderResolver(Parameters const &parameters) : distance_bounds(), materials() {
        float dx = parameters.dx;
        float w = parameters.grid_range()[0];

        float radius = 0.0;
        for (uindex_t i = 0; i < max_n_rings + 1; i++) {
            if (i < parameters.rings.size()) {
                Parameters::RingParameter const &ring = parameters.rings[i];
                radius += ring.width;

                distance_bounds[i] = 2 * (radius / dx) * (radius / dx) - (w * w);
                materials[i] = CoefMaterial::from_relative_material(ring.material, parameters.dx,
                                                                    parameters.dt());
            } else {
                distance_bounds[i] = std::numeric_limits<uindex_t>::max();
                materials[i] = CoefMaterial::perfect_metal();
            }
        }
    }

    CoefMaterial get_material_coefficients(Stencil<MaterialCell, 1> const &stencil) const {
        uindex_t c = stencil.id.c;
        uindex_t r = stencil.id.r;
        uindex_t w = stencil.grid_range.c;
        uindex_t distance_score = 2 * (c * (c - w) + r * (r - w));

#pragma unroll
        for (uindex_ring_t i = 0; i < max_n_rings + 1; i++) {
            if (distance_score < distance_bounds[i]) {
                return materials[i];
            }
        }
        return materials[max_n_rings];
    }

  private:
    uindex_t distance_bounds[max_n_rings + 1];
    CoefMaterial materials[max_n_rings + 1];
};
