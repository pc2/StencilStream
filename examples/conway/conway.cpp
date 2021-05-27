/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <StencilStream/StencilExecutor.hpp>
#include <fstream>

using namespace stencil;
using namespace cl::sycl;

using Cell = bool;
const uindex_t stencil_radius = 1;

buffer<Cell, 2> read(uindex_t width, uindex_t height)
{
    buffer<Cell, 2> input_buffer(range<2>(width, height));
    auto buffer_ac = input_buffer.get_access<access::mode::write>();

    for (uindex_t r = 0; r < height; r++)
    {
        for (uindex_t c = 0; c < width; c++)
        {
            char Cell;
            std::cin >> Cell;
            assert(Cell == 'X' || Cell == '.');
            buffer_ac[c][r] = Cell == 'X';
        }
    }

    return input_buffer;
}

void write(buffer<Cell, 2> output_buffer)
{
    auto buffer_ac = output_buffer.get_access<access::mode::read>();

    uindex_t width = output_buffer.get_range()[0];
    uindex_t height = output_buffer.get_range()[1];

    for (uindex_t r = 0; r < height; r++)
    {
        for (uindex_t c = 0; c < width; c++)
        {
            if (buffer_ac[c][r])
            {
                std::cout << "X";
            }
            else
            {
                std::cout << ".";
            }
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <n_generations>" << std::endl;
        return 1;
    }

    uindex_t width = std::stoi(argv[1]);
    uindex_t height = std::stoi(argv[2]);
    uindex_t n_generations = std::stoi(argv[3]);

    const auto conway = [=](Stencil<Cell, stencil_radius> const &stencil, StencilInfo const &info) {
        ID idx = info.center_cell_id;

        if (idx.c < 0 || idx.r < 0 || idx.c >= width || idx.r >= height)
        {
            // Edge handling, cells in the halo always have to be dead.
            return false;
        }
        else
        {
            uint8_t alive_neighbours = 0;
#pragma unroll
            for (index_t c = -stencil_radius; c <= index_t(stencil_radius); c++)
            {
#pragma unroll
                for (index_t r = -stencil_radius; r <= index_t(stencil_radius); r++)
                {
                    if (stencil[ID(c, r)] && !(c == 0 && r == 0))
                    {
                        alive_neighbours += 1;
                    }
                }
            }

            if (stencil[ID(0,0)])
            {
                return alive_neighbours == 2 || alive_neighbours == 3;
            }
            else
            {
                return alive_neighbours == 3;
            }
        }
    };

    buffer<Cell, 2> grid_buffer = read(width, height);


    using Executor = StencilExecutor<Cell, stencil_radius, decltype(conway)>;
    Executor executor(grid_buffer, false, conway);

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