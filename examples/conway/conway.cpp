/*
 * Copyright © 2020-2023 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include <StencilStream/monotile/Grid.hpp>
#include <StencilStream/monotile/StencilUpdate.hpp>
#include <StencilStream/tdv/NoneSupplier.hpp>

struct ConwayKernel {
    using Cell = bool;
    using TimeDependentValue = std::monostate;

    static constexpr stencil::uindex_t stencil_radius = 1;
    static constexpr stencil::uindex_t n_subgenerations = 1;

    bool operator()(stencil::Stencil<Cell, stencil_radius> const &stencil) const {
        stencil::ID idx = stencil.id;

        uint8_t alive_neighbours = 0;
#pragma unroll
        for (stencil::index_t c = -1; c <= 1; c++) {
#pragma unroll
            for (stencil::index_t r = -1; r <= 1; r++) {
                if (stencil[stencil::ID(c, r)] && !(c == 0 && r == 0)) {
                    alive_neighbours += 1;
                }
            }
        }

        if (stencil[stencil::ID(0, 0)]) {
            return alive_neighbours == 2 || alive_neighbours == 3;
        } else {
            return alive_neighbours == 3;
        }
    }
};

stencil::monotile::Grid<bool> read(stencil::uindex_t width, stencil::uindex_t height) {
    stencil::monotile::Grid<bool> input_grid(width, height);
    {
        auto grid_ac = input_grid.get_access<cl::sycl::access::mode::discard_write>();

        for (stencil::uindex_t r = 0; r < height; r++) {
            for (stencil::uindex_t c = 0; c < width; c++) {
                char Cell;
                std::cin >> Cell;
                assert(Cell == 'X' || Cell == '.');
                grid_ac.set(c, r, Cell == 'X');
            }
        }
    }
    return input_grid;
}

void write(stencil::monotile::Grid<bool> output_grid) {
    auto grid_ac = output_grid.get_access<cl::sycl::access::mode::read>();

    for (stencil::uindex_t r = 0; r < output_grid.get_grid_height(); r++) {
        for (stencil::uindex_t c = 0; c < output_grid.get_grid_width(); c++) {
            if (grid_ac.get(c, r)) {
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
        std::cerr << "Usage: " << argv[0] << " <width> <height> <n_generations>" << std::endl;
        return 1;
    }

    stencil::uindex_t width = std::stoi(argv[1]);
    stencil::uindex_t height = std::stoi(argv[2]);
    stencil::uindex_t n_generations = std::stoi(argv[3]);

    stencil::monotile::Grid<bool> grid = read(width, height);

    stencil::monotile::StencilUpdate<ConwayKernel> update({
        .transition_function = ConwayKernel(),
        .n_generations = n_generations,
    });
    grid = update(grid);

    write(grid);

    return 0;
}