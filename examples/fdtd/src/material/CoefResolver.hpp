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

class CoefResolver {
  public:
    struct MaterialCell : public Cell {
        float ca, cb, da, db;

        MaterialCell() : Cell(), ca(0.0), cb(0.0), da(0.0), db(0.0) {}

        static MaterialCell halo() { return MaterialCell(); };

        static MaterialCell from_parameters(Parameters const &parameters, size_t ring_index) {
            if (ring_index >= parameters.rings.size()) {
                return MaterialCell();
            } else {
                CoefMaterial coefCell = CoefMaterial::from_relative_material(
                    parameters.rings[ring_index].material, parameters.dx, parameters.dt());
                MaterialCell cell = MaterialCell();
                cell.ca = coefCell.ca;
                cell.cb = coefCell.cb;
                cell.da = coefCell.da;
                cell.db = coefCell.db;

                return cell;
            }
        }

        static constexpr auto fields = std::make_tuple(
            &MaterialCell::ex, &MaterialCell::ey, &MaterialCell::hz, &MaterialCell::hz_sum,
            &MaterialCell::ca, &MaterialCell::cb, &MaterialCell::da, &MaterialCell::db);
    };

    CoefResolver(Parameters const &parameters) {}

    CoefMaterial get_material_coefficients(Stencil<MaterialCell, 1, float> const &stencil,
                                           float distance_score) const {
        CoefMaterial cell{.ca = stencil[0][0].ca,
                          .cb = stencil[0][0].cb,
                          .da = stencil[0][0].da,
                          .db = stencil[0][0].db};
        return cell;
    }
};
