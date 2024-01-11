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
#include "../defines.hpp"
#include <cmath>

struct RelMaterial {
    float mu_r, eps_r, sigma;

    static RelMaterial perfect_metal() {
        return RelMaterial{std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::infinity(), 0.0};
    }

    // permeability (transmissibility of the material for magnetic fields) in vacuum
    static constexpr float mu_0 = 4.0 * pi * 1.0e-7;

    // permittivity (transmissibility of the material for electric fields) in vacuum
    static constexpr float eps_0 = 1.0 / (c0 * c0 * mu_0);

    float ca(float dx, float dt) const { return (1 - (sigma * dt)) / (1 + (sigma * dt)); }

    float cb(float dx, float dt) const {
        if (cl::sycl::isinf(eps_r)) {
            return 0.0;
        } else {
            return (dt / (eps_0 * eps_r * dx)) / (1 + (sigma * dt) / (2 * eps_0 * eps_r));
        }
    }

    float da(float dx, float dt) const { return (1 - (sigma * dt)) / (1 + (sigma * dt)); }

    float db(float dx, float dt) const {
        if (cl::sycl::isinf(mu_r)) {
            return 0.0;
        } else {
            return (dt / (mu_0 * mu_r * dx)) / (1 + (sigma * dt) / (2 * mu_0 * mu_r));
        }
    }
};

// The coefficients that describe the properties of a material.
struct CoefMaterial {
    float ca;
    float cb;
    float da;
    float db;

    static CoefMaterial perfect_metal() { return CoefMaterial{1.0, 0.0, 1.0, 0.0}; }

    static CoefMaterial from_relative_material(RelMaterial material, float dx, float dt) {
        return CoefMaterial{material.ca(dx, dt), material.cb(dx, dt), material.da(dx, dt),
                            material.db(dx, dt)};
    }
};