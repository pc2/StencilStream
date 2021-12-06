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

class CoefResolver {
  public:
    using MaterialIdentifier = CoefMaterial;

    CoefResolver(Parameters const &parameters) {}

    CoefMaterial identifier_to_material(MaterialIdentifier identifier) const { return identifier; }

    MaterialIdentifier index_to_identifier(Parameters const &parameters, uindex_t material_index) {
        if (material_index == 0) {
            return CoefMaterial::from_relative(RelMaterial{std::numeric_limits<float>::infinity(),
                                                           std::numeric_limits<float>::infinity(),
                                                           0.0},
                                               parameters.dx, parameters.dt());
        } else {
            return CoefMaterial::from_relative(RelMaterial{11.56, 1.0, 0.0}, parameters.dx,
                                               parameters.dt());
        }
    }
};
