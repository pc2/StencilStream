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

class CoefCell : public Cell {
  public:
    CoefCell() : Cell(), material() {}

    CoefCell(Parameters const &parameters, uindex_t material_index, float ex, float ey, float hz,
             float hz_sum)
        : Cell(ex, ey, hz, hz_sum), material() {
        if (material_index == 0) {
            material = CoefMaterial::from_relative(
                RelMaterial{std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity(), 0.0},
                parameters.dx, parameters.dt());
        } else {
            material = CoefMaterial::from_relative(
                RelMaterial{parameters.mu_r, parameters.eps_r, parameters.sigma}, parameters.dx,
                parameters.dt());
        }
    }

    CoefMaterial material;
};

class CoefResolver {
  public:
    using CellImpl = CoefCell;

    CoefResolver(Parameters const &parameters) {}

    CoefMaterial get_material(CellImpl cell) const { return cell.material; }
};
