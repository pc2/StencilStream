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
#include <StencilStream/BaseTransitionFunction.hpp>
#include <StencilStream/Stencil.hpp>

constexpr int n_main_arguments = 5;
void print_usage(int argc, char **argv);

struct HardwareConfig {
    size_t temporal_parallelism;
    size_t spatial_parallelism;
    size_t cache_width;
};

using namespace stencil;
struct Jacobi1General : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr HardwareConfig mono_config = {
        .temporal_parallelism = 172,
        .spatial_parallelism = 16,
        .cache_width = 8 * 1024,
    };
    static constexpr HardwareConfig multi_mono_config = {
        .temporal_parallelism = 328,
        .spatial_parallelism = 8,
        .cache_width = 4 * 1024,
    };
    static constexpr HardwareConfig tiling_config = {
        .temporal_parallelism = 127,
        .spatial_parallelism = 16,
        .cache_width = 8 * 1024,
    };
    static constexpr size_t n_operations = 1;
    static constexpr size_t n_coefficients = 1;

    float coef;

    Jacobi1General(int argc, char **argv) : coef() {
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

    static constexpr HardwareConfig mono_config = {
        .temporal_parallelism = 104,
        .spatial_parallelism = 16,
        .cache_width = 8 * 1024,
    };
    static constexpr HardwareConfig multi_mono_config = {
        .temporal_parallelism = 188,
        .spatial_parallelism = 8,
        .cache_width = 4 * 1024,
    };
    static constexpr HardwareConfig tiling_config = {
        .temporal_parallelism = 104,
        .spatial_parallelism = 16,
        .cache_width = 8 * 1024,
    };
    static constexpr size_t n_operations = 2;
    static constexpr size_t n_coefficients = 0;

    Jacobi2Constant(int argc, char **argv) {
        if (argc != n_main_arguments)
            print_usage(argc, argv);
    }

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 2 Ops / 4 Bytes
        return (stencil[-1][0] + stencil[1][0]) * 0.5f;
    }
};

struct Jacobi3Constant : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr HardwareConfig mono_config = {
        .temporal_parallelism = 72,
        .spatial_parallelism = 16,
        .cache_width = 16 * 1024,
    };
    static constexpr HardwareConfig multi_mono_config = {
        .temporal_parallelism = 144,
        .spatial_parallelism = 8,
        .cache_width = 8 * 1024,
    };
    static constexpr HardwareConfig tiling_config = {
        .temporal_parallelism = 68,
        .spatial_parallelism = 16,
        .cache_width = 16 * 1024,
    };
    static constexpr size_t n_operations = 3;
    static constexpr size_t n_coefficients = 0;

    Jacobi3Constant(int argc, char **argv) {
        if (argc != n_main_arguments)
            print_usage(argc, argv);
    }

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 3 Ops / 4 Bytes
        return (stencil[-1][0] + stencil[0][0] + stencil[1][0]) * 0.33333334f;
    }
};

struct Jacobi4Constant : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr HardwareConfig mono_config = {
        .temporal_parallelism = 76,
        .spatial_parallelism = 16,
        .cache_width = 16 * 1024,
    };
    static constexpr HardwareConfig multi_mono_config = {
        .temporal_parallelism = 120,
        .spatial_parallelism = 8,
        .cache_width = 8 * 1024,
    };
    static constexpr HardwareConfig tiling_config = {
        .temporal_parallelism = 64,
        .spatial_parallelism = 16,
        .cache_width = 16 * 1024,
    };
    static constexpr size_t n_operations = 4;
    static constexpr size_t n_coefficients = 0;

    Jacobi4Constant(int argc, char **argv) {
        if (argc != n_main_arguments)
            print_usage(argc, argv);
    }

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 4 Ops / 4 Bytes
        return (stencil[-1][0] + stencil[0][-1] + stencil[1][0] + stencil[0][1]) * 0.25f;
    }
};

struct Jacobi5Constant : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr HardwareConfig mono_config = {
        .temporal_parallelism = 56,
        .spatial_parallelism = 16,
        .cache_width = 16 * 1024,
    };
    static constexpr HardwareConfig multi_mono_config = {
        .temporal_parallelism = 120,
        .spatial_parallelism = 8,
        .cache_width = 8 * 1024,
    };
    static constexpr HardwareConfig tiling_config = {
        .temporal_parallelism = 53,
        .spatial_parallelism = 16,
        .cache_width = 16 * 1024,
    };
    static constexpr size_t n_operations = 5;
    static constexpr size_t n_coefficients = 0;

    Jacobi5Constant(int argc, char **argv) {
        if (argc != n_main_arguments)
            print_usage(argc, argv);
    }

    float operator()(stencil::Stencil<float, 1> const &stencil) const {
        // 5 Ops / 4 Bytes
        return (stencil[-1][0] + stencil[0][-1] + stencil[1][0] + stencil[0][1] + stencil[0][0]) *
               0.2f;
    }
};

struct Jacobi4General : public stencil::BaseTransitionFunction {
    using Cell = float;

    static constexpr HardwareConfig mono_config = {
        .temporal_parallelism = 52,
        .spatial_parallelism = 16,
        .cache_width = 16 * 1024,
    };
    static constexpr HardwareConfig multi_mono_config = {
        .temporal_parallelism = 112,
        .spatial_parallelism = 8,
        .cache_width = 8 * 1024,
    };
    static constexpr HardwareConfig tiling_config = {
        .temporal_parallelism = 56,
        .spatial_parallelism = 16,
        .cache_width = 16 * 1024,
    };
    static constexpr size_t n_operations = 7;
    static constexpr size_t n_coefficients = 4;

    Jacobi4General(int argc, char **argv) : coef() {
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

    static constexpr HardwareConfig mono_config = {
        .temporal_parallelism = 28,
        .spatial_parallelism = 16,
        .cache_width = 16 * 1024,
    };
    static constexpr HardwareConfig multi_mono_config = {
        .temporal_parallelism = 100,
        .spatial_parallelism = 8,
        .cache_width = 8 * 1024,
    };
    static constexpr HardwareConfig tiling_config = {
        .temporal_parallelism = 44,
        .spatial_parallelism = 16,
        .cache_width = 16 * 1024,
    };
    static constexpr size_t n_operations = 9;
    static constexpr size_t n_coefficients = 5;

    Jacobi5General(int argc, char **argv) : coef() {
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

    static constexpr HardwareConfig mono_config = {
        .temporal_parallelism = 4,
        .spatial_parallelism = 16,
        .cache_width = 32 * 1024,
    };
    static constexpr HardwareConfig multi_mono_config = {
        .temporal_parallelism = 28,
        .spatial_parallelism = 8,
        .cache_width = 16 * 1024,
    };
    static constexpr HardwareConfig tiling_config = {
        .temporal_parallelism = 32,
        .spatial_parallelism = 16,
        .cache_width = 32 * 1024,
    };
    static constexpr size_t n_operations = 17;
    static constexpr size_t n_coefficients = 9;

    Jacobi9General(int argc, char **argv) : coef() {
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
#if defined(STENCILSTREAM_BACKEND_MONOTILE)
    #if JACOBI_MULTI_FPGA == 1
constexpr HardwareConfig hardware_config = JacobiKernel::multi_mono_config;
    #else
constexpr HardwareConfig hardware_config = JacobiKernel::mono_config;
    #endif
#else
constexpr HardwareConfig hardware_config = JacobiKernel::tiling_config;
#endif

const size_t temporal_parallelism = hardware_config.temporal_parallelism;
const size_t spatial_parallelism = hardware_config.spatial_parallelism;
const size_t tile_height = 1 << 16;
#if defined(STENCILSTREAM_BACKEND_MONOTILE)
const size_t tile_width = hardware_config.cache_width;
#elif defined(STENCILSTREAM_BACKEND_TILING)
const size_t tile_width =
    hardware_config.cache_width - spatial_parallelism * (2 * temporal_parallelism + 1);
#endif

const size_t n_kernels = stencil::internal::int_ceil_div<size_t>(temporal_parallelism, 4);
