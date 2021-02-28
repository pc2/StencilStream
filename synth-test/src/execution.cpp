/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <ExecutionPipeline.hpp>

using namespace cl::sycl;
using namespace stencil;

const UIndex radius = 2;
const UIndex width = 1024;
const UIndex height = 1024;
const UIndex pipeline_length = 1;

int main()
{

#ifdef HARDWARE
    INTEL::fpga_selector device_selector;
#else
    INTEL::fpga_emulator_selector device_selector;
#endif
    queue fpga_queue(device_selector);

    buffer<ID, 2> out_buffer(range<2>(width, height));

    auto kernel = [](Stencil<ID, radius> const &stencil, StencilInfo const &info) {
        UIndex center_column = info.center_cell_id.c;
        UIndex center_row = info.center_cell_id.r;

        bool is_valid = true;
#pragma unroll
        for (Index c = -Index(radius); c <= Index(radius); c++)
        {
#pragma unroll
            for (Index r = -Index(radius); r <= Index(radius); r++)
            {
                is_valid &= stencil[ID(c, r)].c == Index(c + center_column);
                is_valid &= stencil[ID(c, r)].r == Index(r + center_row);
            }
        }

        if (is_valid)
        {
            return stencil[ID(0, 0)];
        }
        else
        {
            return ID(0, 0);
        }
    };

    fpga_queue.submit([&](handler &cgh) {
        auto out_buffer_ac = out_buffer.get_access<access::mode::discard_write>(cgh);

        cgh.single_task([=]() {
            ExecutionPipeline<ID, radius, pipeline_length, width, height, decltype(kernel)> pipeline(0, 0, 0, kernel);
            //ExecutionCore<ID, radius, 2*radius + height> pipeline(0,width,height,0,0);

            Index in_c, in_r, out_c, out_r;
            in_c = in_r = -(radius * pipeline_length);
            out_c = out_r = 0;
            UIndex n_input_cells = (width + 2 * radius * pipeline_length) * (height + 2 * radius * pipeline_length);

            for (UIndex i_cell = 0; i_cell < n_input_cells; i_cell++)
            {
                ID input(in_c, in_r);
                if (in_r == height + radius * pipeline_length - 1)
                {
                    in_r = -(radius * pipeline_length);
                    in_c++;
                }
                else
                {
                    in_r++;
                }

                std::optional<ID> next_output = pipeline.step(input);
                //std::optional<ID> next_output = pipeline.template step<decltype(kernel)>(input, kernel);

                if (next_output.has_value())
                {
                    out_buffer_ac[out_c][out_r] = *next_output;
                    if (out_r == height - 1)
                    {
                        out_r = 0;
                        out_c += 1;
                    }
                    else
                    {
                        out_r += 1;
                    }
                }
            }
        });
    });

    auto out_buffer_ac = out_buffer.get_access<access::mode::read>();
    for (UIndex c = 0; c < width; c++)
    {
        for (UIndex r = 0; r < height; r++)
        {
            assert(out_buffer_ac[c][r].c == c);
            assert(out_buffer_ac[c][r].r == r);
        }
    }

    return 0;
}