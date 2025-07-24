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

#include <StencilStream/BaseTransitionFunction.hpp>
#include <StencilStream/cpu/StencilUpdate.hpp>
#include <StencilStream/cuda/StencilUpdate.hpp>
#include <StencilStream/monotile/StencilUpdate.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace stencil;
#if defined(STENCILSTREAM_BACKEND_CPU)
using namespace stencil::cpu;
#elif defined(STENCILSTREAM_BACKEND_CUDA)
using namespace stencil::cuda;
#else
using namespace stencil::monotile;
#endif

struct ConwayKernel : public BaseTransitionFunction {
    using Cell = bool;

    bool operator()(Stencil<bool, stencil_radius> const &stencil) const {
        int alive_neighbours = 0;
#pragma unroll
        for (int r = -1; r <= 1; r++) {
#pragma unroll
            for (int c = -1; c <= 1; c++) {
                if (stencil[r][c] && !(r == 0 && c == 0)) {
                    alive_neighbours += 1;
                }
            }
        }

        if (stencil[0][0]) {
            return alive_neighbours == 2 || alive_neighbours == 3;
        } else {
            return alive_neighbours == 3;
        }
    }
};

Grid<bool> read(std::size_t height, std::size_t width) {
    Grid<bool> input_grid(height, width);
    {
        Grid<bool>::GridAccessor<sycl::access::mode::read_write> grid_ac(input_grid);

        for (std::size_t r = 0; r < height; r++) {
            for (std::size_t c = 0; c < width; c++) {
                char cell;
                std::cin >> cell;
                assert(cell == 'X' || cell == '.');
                grid_ac[r][c] = cell == 'X';
            }
        }
    }
    return input_grid;
}

void write(Grid<bool> output_grid) {
    Grid<bool>::GridAccessor<sycl::access::mode::read> grid_ac(output_grid);

    for (std::size_t r = 0; r < output_grid.get_grid_height(); r++) {
        for (std::size_t c = 0; c < output_grid.get_grid_width(); c++) {
            if (grid_ac[r][c]) {
                std::cout << "X";
            } else {
                std::cout << ".";
            }
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <height> <width> <n_iterations>" << std::endl;
        return 1;
    }

    std::size_t height = std::stoi(argv[1]);
    std::size_t width = std::stoi(argv[2]);
    std::size_t n_iterations = std::stoi(argv[3]);

    Grid<bool> grid = read(height, width);

#if defined(STENCILSTREAM_TARGET_FPGA)
    sycl::device device(sycl::ext::intel::fpga_selector_v);
#elif defined(STENCILSTREAM_TARGET_CUDA)
    sycl::device device(sycl::gpu_selector_v);
#else
    sycl::device device;
#endif

    StencilUpdate<ConwayKernel> update({
        .transition_function = ConwayKernel(),
        .n_iterations = n_iterations,
#if !defined(STENCILSTREAM_BACKEND_MONOTILE)
        .device = device,
#endif
    });
    grid = update(grid);

    write(grid);

    return 0;
}