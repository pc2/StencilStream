/*
 * Copyright © 2020 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "stencil/stencil.hpp"
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <fstream>

using cell = bool;
const stencil::uindex_t radius = 1;
const stencil::uindex_t width = 64;
const stencil::uindex_t height = 64;

const auto conway = [](stencil::Stencil2D<cell, radius> const &stencil, stencil::Stencil2DInfo const &info) {
    stencil::UID idx = info.center_cell_id;

    stencil::index_t lower_h_bound, lower_v_bound;
    lower_h_bound = lower_v_bound = -radius;
    stencil::index_t upper_h_bound, upper_v_bound;
    upper_h_bound = upper_v_bound = radius;

    if (idx.c == 0)
    {
        lower_h_bound = 0;
    }
    else if (idx.c == width - 1)
    {
        upper_h_bound = 0;
    }

    if (idx.r == 0)
    {
        lower_v_bound = 0;
    }
    else if (idx.r == height - 1)
    {
        upper_v_bound = 0;
    }

    uint8_t alive_neighbours = 0;
#pragma unroll
    for (stencil::index_t c = -radius; c <= stencil::index_t(radius); c++)
    {
#pragma unroll
        for (stencil::index_t r = -radius; r <= stencil::index_t(radius); r++)
        {
            bool is_valid_neighbour = (c != 0 || r != 0) &&
                                      (c >= lower_h_bound) &&
                                      (c <= upper_h_bound) &&
                                      (r >= lower_v_bound) &&
                                      (r <= upper_v_bound);

            if (is_valid_neighbour && stencil[stencil::ID(c, r)])
            {
                alive_neighbours += 1;
            }
        }
    }

    if (stencil[stencil::ID(0, 0)])
    {
        return alive_neighbours == 2 || alive_neighbours == 3;
    }
    else
    {
        return alive_neighbours == 3;
    }
};

cl::sycl::buffer<cell, 2> read()
{
    cl::sycl::buffer<cell, 2> cell_buffer(cl::sycl::range<2>(width, height));
    auto buffer_ac = cell_buffer.get_access<cl::sycl::access::mode::write>();
    std::fstream input_file("input.txt", std::ios::in);

    for (stencil::uindex_t r = 0; r < height; r++)
    {
        for (stencil::uindex_t c = 0; c < width; c++)
        {
            char cell;
            input_file >> cell;
            buffer_ac[c][r] = cell == 'X';
        }
    }

    return cell_buffer;
}

void write(cl::sycl::buffer<cell, 2> cell_buffer)
{
    assert(cell_buffer.get_range() == cl::sycl::range<2>(width, height));
    auto buffer_ac = cell_buffer.get_access<cl::sycl::access::mode::read>();
    std::fstream output_file("output.txt", std::ios::out);

    for (stencil::uindex_t r = 0; r < height; r++)
    {
        for (stencil::uindex_t c = 0; c < width; c++)
        {
            if (buffer_ac[c][r])
            {
                output_file << "X";
            }
            else
            {
                output_file << ".";
            }
        }
        output_file << std::endl;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <#Generations>" << std::endl;
        return 1;
    }

#ifdef HARDWARE
    cl::sycl::INTEL::fpga_selector device_selector;
#else
    cl::sycl::INTEL::fpga_emulator_selector device_selector;
#endif
    cl::sycl::queue fpga_queue(device_selector);

    cl::sycl::buffer<cell, 2> cell_buffer = read();

    stencil::StencilExecutor<cell, radius, width, height> executor(fpga_queue);
    executor.set_buffer(cell_buffer);
    executor.set_generations(std::stoi(argv[1]));
    executor.run(conway);

    write(cell_buffer);

    return 0;
}