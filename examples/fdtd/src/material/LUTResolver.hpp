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
#include <StencilStream/Stencil.hpp>

class LUTResolver {
  public:
    struct MaterialCell {
        Cell cell;
        uindex_ring_t index;

        static MaterialCell halo() { return MaterialCell{Cell::halo(), 0}; }

        static MaterialCell from_parameters(Parameters const &parameters, uindex_t ring_index) {
            return MaterialCell{Cell::halo(), uindex_ring_t(ring_index)};
        }
    };

    LUTResolver(Parameters const &parameters) : materials() {
        for (uindex_t i = 0; i < max_n_rings + 1; i++) {
            if (i < parameters.rings.size()) {
                materials[i] = CoefMaterial::from_relative_material(parameters.rings[i].material,
                                                                    parameters.dx, parameters.dt());
            } else {
                materials[i] = CoefMaterial::perfect_metal();
            }
        }
    }

    CoefMaterial get_material_coefficients(Stencil<MaterialCell, 1, float> const &stencil,
                                           index_t distance_score) const {
        return materials[stencil[ID(0, 0)].index];
    }

  private:
    CoefMaterial materials[max_n_rings + 1];
};
