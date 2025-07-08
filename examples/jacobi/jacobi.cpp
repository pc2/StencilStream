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
#include "kernels.hpp"

#if defined(STENCILSTREAM_BACKEND_MONOTILE)
    #include <StencilStream/monotile/StencilUpdate.hpp>
const char *variant = "monotile";
#elif defined(STENCILSTREAM_BACKEND_TILING)
    #include <StencilStream/tiling/StencilUpdate.hpp>
const char *variant = "tiling";
#endif
#include <fstream>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace stencil;

#if defined(STENCILSTREAM_BACKEND_MONOTILE)
using StencilUpdate =
    monotile::StencilUpdate<JacobiKernel, temporal_parallelism, spatial_parallelism, tile_height,
                            tile_width, n_kernels>;

#elif defined(STENCILSTREAM_BACKEND_CPU)
using StencilUpdate = cpu::StencilUpdate<JacobiKernel>;

#elif defined(STENCILSTREAM_BACKEND_TILING)
using StencilUpdate = tiling::StencilUpdate<JacobiKernel, temporal_parallelism, spatial_parallelism,
                                            tile_height, tile_width, n_kernels>;
#endif

using Grid = StencilUpdate::GridImpl;

void print_usage(int argc, char **argv) {
    std::cerr << "Usage: " << argv[0]
              << "  <grid_rows> <grid_cols> <no. of iterations> <output_file> <coef>" << std::endl;
    std::cerr << "    <grid_rows>         - number of rows in the grid (positive integer)"
              << std::endl;
    std::cerr << "    <grid_cols>         - number of columns in the grid (positive integer)"
              << std::endl;
    std::cerr << "    <no. of iterations> - number of iterations (positive integer)" << std::endl;
    std::cerr << "    <output_file>       - path to the output file" << std::endl;
    std::cerr
        << "    <coef>              - coefficients for general variants (floating-point numbers)"
        << std::endl;

    exit(1);
}

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "show-config") == 0) {
        std::cout << "{" << std::endl;
        std::cout << "    \"variant\": \"" << variant << "\"," << std::endl;
        std::cout << "    \"temporal_parallelism\": " << temporal_parallelism << "," << std::endl;
        std::cout << "    \"spatial_parallelism\": " << spatial_parallelism << "," << std::endl;
        std::cout << "    \"tile_height\": " << tile_height << "," << std::endl;
        std::cout << "    \"tile_width\": " << tile_width << "," << std::endl;
        std::cout << "    \"n_coefficients\": " << JacobiKernel::n_coefficients << "," << std::endl;
        std::cout << "    \"n_operations\": " << JacobiKernel::n_operations << std::endl;
        std::cout << "}" << std::endl;
        exit(0);
    } else if (argc < n_main_arguments) {
        print_usage(argc, argv);
    }

    sycl::range<2> range(atoi(argv[1]), atoi(argv[2]));
    size_t n_iterations = atoi(argv[3]);
    std::string out_path(argv[4]);

    Grid grid(range);
    {
        Grid::GridAccessor<sycl::access::mode::read_write> grid_ac(grid);
        for (size_t r = 0; r < range[0]; r++) {
            for (size_t c = 0; c < range[1]; c++) {
                if (r >= range[0] * 0.25 && r < range[0] * 0.75 && c >= range[1] * 0.25 &&
                    c < range[1] * 0.75) {
                    grid_ac[r][c] = 1.0;
                } else {
                    grid_ac[r][c] = 0.0;
                }
            }
        }
    }

#if defined(STENCILSTREAM_TARGET_FPGA)
    sycl::device device(sycl::ext::intel::fpga_selector_v);
#else
    sycl::device device;
#endif

    StencilUpdate update({
        .transition_function = JacobiKernel(argc, argv),
        .halo_value = 0.0,
        .n_iterations = n_iterations,
        .device = device,
        .blocking = true,
    });

    std::cout << "Starting simulation" << std::endl;

    grid = update(grid);

    std::cout << "Simulation complete!" << std::endl;
    std::cout << "Walltime: " << update.get_walltime() << " s" << std::endl;

    {
        Grid::GridAccessor<sycl::access::mode::read> grid_ac(grid);
        std::fstream out(out_path, out.out | out.trunc | out.binary);

        if (!out.is_open()) {
            throw std::runtime_error("The output file can't be opened!\n");
        }

        for (size_t r = 0; r < range[0]; r++) {
            for (size_t c = 0; c < range[1]; c++) {
                out.write((char *)&grid_ac[r][c], sizeof(float));
            }
        }
    }

    return 0;
}