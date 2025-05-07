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
#include <StencilStream/Stencil.hpp>
#include <StencilStream/BaseTransitionFunction.hpp>

constexpr int n_main_arguments = 5;
void print_usage(int argc, char** argv) {
    std::cerr << "Usage: " << argv[0]
            << "  <grid_rows> <grid_cols> <no. of iterations> <output_file> <coef>"
            << std::endl;
    std::cerr << "    <grid_rows>         - number of rows in the grid (positive integer)"
            << std::endl;
    std::cerr << "    <grid_cols>         - number of columns in the grid (positive integer)"
            << std::endl;
    std::cerr << "    <no. of iterations> - number of iterations (positive integer)" << std::endl;
    std::cerr << "    <output_file>       - path to the output file" << std::endl;
    std::cerr << "    <coef>              - coefficients for general variants (floating-point numbers)" << std::endl;
    exit(1);
}

using namespace stencil;
struct Jacobi1General : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr size_t temporal_parallelism = 544;
    static constexpr size_t mono_tile_width = 64 * 1024;

    float coef;

    Jacobi1General(int argc, char** argv) : coef() {
        if (argc != n_main_arguments + 1)
            print_usage(argc, argv);
        coef = atof(argv[n_main_arguments]);
    }

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 1 Ops / 4 Bytes
        return coef * stencil[0][0];
    }
};

struct Jacobi2Constant : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr size_t temporal_parallelism = 272;
    static constexpr size_t mono_tile_width = 32 * 1024;

    Jacobi2Constant(int argc, char** argv) {
        if (argc != n_main_arguments)
            print_usage(argc, argv);
    }

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 2 Ops / 4 Bytes
        return (stencil[-1][0] + stencil[1][0]) / 2.0;
    }
};

struct Jacobi3Constant : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr size_t temporal_parallelism = 181;
    static constexpr size_t mono_tile_width = 32 * 1024;

    Jacobi3Constant(int argc, char** argv) {
        if (argc != n_main_arguments)
            print_usage(argc, argv);
    }

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 3 Ops / 4 Bytes
        return (stencil[-1][0] + stencil[0][0] + stencil[1][0]) / 3.0;
    }
};

struct Jacobi4Constant : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr size_t temporal_parallelism = 136;
    static constexpr size_t mono_tile_width = 32 * 1024;

    Jacobi4Constant(int argc, char** argv) {
        if (argc != n_main_arguments)
            print_usage(argc, argv);
    }

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 4 Ops / 4 Bytes
        return (stencil[-1][0] + stencil[0][-1] + stencil[1][0] + stencil[0][-1]) / 4.0;
    }
};

struct Jacobi5Constant : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr size_t temporal_parallelism = 88;
    static constexpr size_t mono_tile_width = 12 * 1024;

    Jacobi5Constant(int argc, char** argv) {
        if (argc != n_main_arguments)
            print_usage(argc, argv);
    }

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 5 Ops / 4 Bytes
        return (stencil[-1][0] + stencil[0][-1] + stencil[1][0] + stencil[0][-1] + stencil[0][0]) /
               5.0;
    }
};

struct Jacobi4General : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr size_t temporal_parallelism = 72;
    static constexpr size_t mono_tile_width = 16 * 1024;

    Jacobi4General(int argc, char** argv) : coef() {
        if (argc != n_main_arguments + 4)
            print_usage(argc, argv);
        for (int i = 0; i < 4; i++) {
            coef[i] = atof(argv[n_main_arguments + i]);
        }
    }

    float coef[4];

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 7 Ops / 4 Bytes
        return coef[0] * stencil[-1][0] + coef[1] * stencil[0][-1] + coef[2] * stencil[1][0] +
               coef[3] * stencil[0][1];
    }
};

struct Jacobi5General : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr size_t temporal_parallelism = 56;
    static constexpr size_t mono_tile_width = 16 * 1024;

    Jacobi5General(int argc, char** argv) : coef() {
        if (argc != n_main_arguments + 5)
            print_usage(argc, argv);
        for (int i = 0; i < 5; i++) {
            coef[i] = atof(argv[n_main_arguments + i]);
        }
    }

    float coef[5];

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 9 Ops / 4 Bytes
        return coef[0] * stencil[-1][0] + coef[1] * stencil[0][-1] + coef[2] * stencil[1][0] +
               coef[3] * stencil[0][1] + coef[4] * stencil[0][0];
    }
};

struct Jacobi9General : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr size_t temporal_parallelism = 32;
    static constexpr size_t mono_tile_width = 32 * 1024;

    Jacobi9General(int argc, char** argv) : coef() {
        if (argc != n_main_arguments + 9)
            print_usage(argc, argv);
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                coef[r][c] = atof(argv[n_main_arguments + r * 3 + c]);
            }
        }
    }

    float coef[3][3];

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 17 Ops / 4 Bytes
        float sum = 0.0;
#pragma unroll
        for (int r = -1; r <= 1; r++) {
#pragma unroll
            for (int c = -1; c <= 1; c++) {
                sum += coef[r + 1][c + 1] * stencil[r][c];
            }
        }
        return sum;
    }
};

using JacobiKernel = JACOBI_KERNEL;
const size_t temporal_parallelism = JacobiKernel::temporal_parallelism;
const size_t spatial_parallelism = 16;
const size_t tile_height = 1 << 16;
const size_t mono_tile_width = JacobiKernel::mono_tile_width;
const size_t tiling_tile_width = mono_tile_width - spatial_parallelism * (2 * temporal_parallelism + 1);
const size_t n_kernels = temporal_parallelism / 4;
