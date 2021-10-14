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
#include "../res/TransFuncs.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <StencilStream/StencilExecutor.hpp>

using namespace std;
using namespace cl::sycl;
using namespace stencil;

const uindex_t stencil_radius = 1;
const uindex_t pipeline_length = 34;
const uindex_t tile_width = 1024;
const uindex_t tile_height = 1024;
const uindex_t grid_width = 2 * tile_width;
const uindex_t grid_height = 2 * tile_height;
const uindex_t burst_size = 1024;

using TransFunc = FPGATransFunc<stencil_radius>;
using Executor = StencilExecutor<Cell, stencil_radius, TransFunc, pipeline_length, tile_width,
                                 tile_height, burst_size>;

void exception_handler(cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (cl::sycl::exception const &e) {
            std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
            std::terminate();
        }
    }
}

int main() {
#ifdef HARDWARE
    ext::intel::fpga_selector device_selector;
#else
    ext::intel::fpga_emulator_selector device_selector;
#endif
    cl::sycl::queue working_queue(device_selector, exception_handler);

    buffer<Cell, 2> in_buffer(range<2>(grid_width, grid_height));
    {
        auto in_buffer_ac = in_buffer.get_access<access::mode::discard_write>();

        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                in_buffer_ac[c][r] = Cell{index_t(c), index_t(r), 0, CellStatus::Normal};
            }
        }
    }

    std::cout << "Input loaded" << std::endl;

    Executor executor(Cell::halo(), TransFunc());
    executor.set_input(in_buffer);
    executor.set_queue(working_queue);

    executor.run(2 * pipeline_length);

    buffer<Cell, 2> out_buffer(range<2>(grid_width, grid_height));
    executor.copy_output(out_buffer);

    std::cout << "Output loaded" << std::endl;

    {
        auto out_buffer_ac = out_buffer.get_access<access::mode::read>();

        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                assert(out_buffer_ac[c][r].c == c);
                assert(out_buffer_ac[c][r].r == r);
                assert(out_buffer_ac[c][r].i_generation == 2 * pipeline_length);
                assert(out_buffer_ac[c][r].status == CellStatus::Normal);
            }
        }
    }

    std::cout << "Test finished successfully!" << std::endl;

    return 0;
}