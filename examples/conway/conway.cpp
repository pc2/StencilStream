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
#include <StencilStream/monotile/StencilUpdate.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace stencil;
#if defined(STENCILSTREAM_BACKEND_CPU)
using namespace stencil::cpu;
#else
using namespace stencil::monotile;
#endif

struct ConwayKernel : public BaseTransitionFunction {
    using Cell = bool;

    bool operator()(Stencil<bool, stencil_radius> const &stencil) const {
        ID idx = stencil.id;

        uint8_t alive_neighbours = 0;
#pragma unroll
        for (index_t c = -1; c <= 1; c++) {
#pragma unroll
            for (index_t r = -1; r <= 1; r++) {
                if (stencil[ID(c, r)] && !(c == 0 && r == 0)) {
                    alive_neighbours += 1;
                }
            }
        }

        if (stencil[ID(0, 0)]) {
            return alive_neighbours == 2 || alive_neighbours == 3;
        } else {
            return alive_neighbours == 3;
        }
    }
};

Grid<bool> read(uindex_t width, uindex_t height) {
    Grid<bool> input_grid(width, height);
    {
        Grid<bool>::GridAccessor<sycl::access::mode::read_write> grid_ac(input_grid);

        for (uindex_t r = 0; r < height; r++) {
            for (uindex_t c = 0; c < width; c++) {
                char cell;
                std::cin >> cell;
                assert(cell == 'X' || cell == '.');
                grid_ac[c][r] = cell == 'X';
            }
        }
    }
    return input_grid;
}

void write(Grid<bool> output_grid) {
    Grid<bool>::GridAccessor<sycl::access::mode::read> grid_ac(output_grid);

    for (uindex_t r = 0; r < output_grid.get_grid_height(); r++) {
        for (uindex_t c = 0; c < output_grid.get_grid_width(); c++) {
            if (grid_ac[c][r]) {
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
        std::cerr << "Usage: " << argv[0] << " <width> <height> <n_iterations>" << std::endl;
        return 1;
    }

    uindex_t width = std::stoi(argv[1]);
    uindex_t height = std::stoi(argv[2]);
    uindex_t n_iterations = std::stoi(argv[3]);

    Grid<bool> grid = read(width, height);

#if defined(STENCILSTREAM_TARGET_FPGA)
    sycl::device device(sycl::ext::intel::fpga_selector_v);
#else
    sycl::device device;
#endif

    StencilUpdate<ConwayKernel> update({
        .transition_function = ConwayKernel(),
        .n_iterations = n_iterations,
        .device = device,
    });
    grid = update(grid);

    write(grid);

    return 0;
}