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
    struct MaterialCell : public Cell {
        uindex_ring_t index;

        MaterialCell() : Cell(), index(0) {}

        static MaterialCell halo() { return MaterialCell(); }

        static MaterialCell from_parameters(Parameters const &parameters, size_t ring_index) {
            MaterialCell cell;
            cell.index = uindex_ring_t(ring_index);
            return cell;
        }

        static constexpr auto fields =
            std::make_tuple(&MaterialCell::ex, &MaterialCell::ey, &MaterialCell::hz,
                            &MaterialCell::hz_sum, &MaterialCell::index);
    };

    LUTResolver(Parameters const &parameters) : materials() {
        for (size_t i = 0; i < max_n_rings + 1; i++) {
            if (i < parameters.rings.size()) {
                materials[i] = CoefMaterial::from_relative_material(parameters.rings[i].material,
                                                                    parameters.dx, parameters.dt());
            } else {
                materials[i] = CoefMaterial::perfect_metal();
            }
        }
    }

    CoefMaterial get_material_coefficients(Stencil<MaterialCell, 1, float> const &stencil,
                                           float distance_score) const {
        return materials[stencil[0][0].index];
    }

  private:
    CoefMaterial materials[max_n_rings + 1];
};
