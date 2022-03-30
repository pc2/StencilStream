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
#include "../Parameters.hpp"
#include "Material.hpp"
#include "../Cell.hpp"

class LUTCell : public Cell {
  public:
    LUTCell() : Cell(), index(0) {}

    LUTCell(Parameters const &parameters, uindex_t material_index, float ex, float ey, float hz,
            float hz_sum)
        : Cell(ex, ey, hz, hz_sum), index(material_index) {}

    uint8_t index;
};

class LUTResolver {
  public:
    using CellImpl = LUTCell;

    LUTResolver(Parameters const &parameters) : materials() {
        materials[0] =
            CoefMaterial::from_relative(RelMaterial{std::numeric_limits<float>::infinity(),
                                                    std::numeric_limits<float>::infinity(), 0.0},
                                        parameters.dx, parameters.dt());
        materials[1] = CoefMaterial::from_relative(
            RelMaterial{parameters.mu_r, parameters.eps_r, parameters.sigma}, parameters.dx,
            parameters.dt());
    }

    CoefMaterial get_material(CellImpl cell) const {
        uint8_t index = cell.index;
        if (index > 1) {
            index = 0;
        }
        return materials[index];
    }

  private:
    CoefMaterial materials[2];
};
