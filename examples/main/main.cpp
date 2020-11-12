/*
 * Copyright © 2020 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <functional>
#include <iostream>
#include <stencil/stencil.hpp>
#include <type_traits>

using namespace cl::sycl;
using namespace stencil;


const uindex_t radius = 1;
const uindex_t max_width = 1024;
const uindex_t max_height = 1024;

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <generations>" << std::endl;
        return 1;
    }

    uindex_t width = std::stoi(argv[1]);
    uindex_t height = std::stoi(argv[2]);
    uindex_t n_generations = std::stoi(argv[3]);

#ifdef HARDWARE
    INTEL::fpga_selector device_selector;
#else
    INTEL::fpga_emulator_selector device_selector;
#endif
    queue fpga_queue(device_selector);

    buffer<int32_t, 2> buffer(range<2>(width, height));
    {
        auto ac = buffer.get_access<access::mode::write>();
        for (int32_t c = 0; c < width; c++)
        {
            for (int32_t r = 0; r < height; r++)
            {
                ac[c][r] = c + r;
            }
        }
    }

    auto kernel = [width, height](Stencil2D<int32_t, radius> const &stencil, Stencil2DInfo const &info) {
        auto idx = info.center_cell_id;

        int32_t middle = idx.c + idx.r;
        bool is_valid = true;
        int32_t value;

        int32_t coef;
        if (idx.c == 0 && idx.r == 0 && stencil[ID(1, 1)] < 0)
        {
            coef = -1;
        }
        else if ((idx.c > 0 || idx.r > 0) && stencil[ID(0, 0)] < 0)
        {
            coef = -1;
        }
        else
        {
            coef = 1;
        }

        value = coef * stencil[ID(-1, -1)];
        is_valid &= idx.c == 0 || idx.r == 0 || value == middle - 2;

        value = coef * stencil[ID(0, -1)];
        is_valid &= idx.r == 0 || value == middle - 1;

        value = coef * stencil[ID(1, -1)];
        is_valid &= idx.c == width - 1 || idx.r == 0 || value == middle;

        value = coef * stencil[ID(-1, 0)];
        is_valid &= idx.c == 0 || value == middle - 1;

        value = coef * stencil[ID(0, 0)];
        is_valid &= value == middle;

        value = coef * stencil[ID(1, 0)];
        is_valid &= idx.c == width - 1 || value == middle + 1;

        value = coef * stencil[ID(-1, 1)];
        is_valid &= idx.c == 0 || idx.r == height - 1 || value == middle;

        value = coef * stencil[ID(0, 1)];
        is_valid &= idx.r == height - 1 || value == middle + 1;

        value = coef * stencil[ID(1, 1)];
        is_valid &= idx.c == width - 1 || idx.r == height - 1 || value == middle + 2;

        if (is_valid)
        {
            return -1 * middle;
        }
        else
        {
            int32_t sum;
#pragma unroll
            for (index_t c = -radius; c <= radius; c++)
            {
#pragma unroll
                for (index_t r = -radius; r <= radius; r++)
                {
                    sum += stencil[ID(c, r)];
                }
            }
            return sum;
        }
    };

    std::cout << "Executing..." << std::endl;
    StencilExecutor<int32_t, 1, max_width, max_height> executor(fpga_queue);
    executor.set_buffer(buffer);
    executor.set_generations(n_generations);
    executor.run(kernel);

    std::cout << "Done executing, waiting for the buffer to become available..." << std::endl;

    auto buffer_ac = buffer.get_access<access::mode::read>();
    bool is_valid = true;
    for (index_t c = 0; c < width; c++)
    {
        for (index_t r = 0; r < height; r++)
        {
            int32_t value = buffer_ac[c][r];
            if (value != -1 * (c + r))
            {
                std::cout << "(" << c << ", " << r << ") => " << value;
                std::cout << " (!= " << (int32_t)(-1 * (c + r)) << ")" << std::endl;
                is_valid = false;
            }
        }
    }
    assert(is_valid);

    return 0;
}