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
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <StencilStream/StencilExecutor.hpp>

using Cell = bool;
const Cell halo_value = false;
const stencil::uindex_t stencil_radius = 1;

auto conway = [](stencil::Stencil<Cell, stencil_radius> const &stencil) {
    stencil::ID idx = stencil.id;

    uint8_t alive_neighbours = 0;
#pragma unroll
    for (stencil::index_t c = -stencil_radius; c <= stencil::index_t(stencil_radius); c++) {
#pragma unroll
        for (stencil::index_t r = -stencil_radius; r <= stencil::index_t(stencil_radius); r++) {
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
};

cl::sycl::buffer<Cell, 2> read(stencil::uindex_t width, stencil::uindex_t height) {
    cl::sycl::buffer<Cell, 2> input_buffer(cl::sycl::range<2>(width, height));
    auto buffer_ac = input_buffer.get_access<cl::sycl::access::mode::write>();

    for (stencil::uindex_t r = 0; r < height; r++) {
        for (stencil::uindex_t c = 0; c < width; c++) {
            char Cell;
            std::cin >> Cell;
            assert(Cell == 'X' || Cell == '.');
            buffer_ac[c][r] = Cell == 'X';
        }
    }

    return input_buffer;
}

void write(cl::sycl::buffer<Cell, 2> output_buffer) {
    auto buffer_ac = output_buffer.get_access<cl::sycl::access::mode::read>();

    stencil::uindex_t width = output_buffer.get_range()[0];
    stencil::uindex_t height = output_buffer.get_range()[1];

    for (stencil::uindex_t r = 0; r < height; r++) {
        for (stencil::uindex_t c = 0; c < width; c++) {
            if (buffer_ac[c][r]) {
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

    cl::sycl::buffer<Cell, 2> grid_buffer = read(width, height);

    using Executor = stencil::StencilExecutor<Cell, stencil_radius, decltype(conway)>;
    Executor executor(halo_value, conway);
    executor.set_input(grid_buffer);

#ifdef HARDWARE
    executor.select_fpga();
#else
    executor.select_emulator();
#endif

    executor.run(n_generations);

    executor.copy_output(grid_buffer);
    write(grid_buffer);

    return 0;
}