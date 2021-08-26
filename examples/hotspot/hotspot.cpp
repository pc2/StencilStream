/*
 * Copyright © 2020-2021Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <StencilStream/MonotileExecutor.hpp>
#include <StencilStream/StencilExecutor.hpp>
#include <fstream>

using namespace std;
using namespace cl::sycl;
using namespace stencil;

typedef float FLOAT;

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD (3.0e6)
/* required precision in degrees	*/
#define PRECISION 0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP 0.5

/* chip parameters	*/
const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 0.016;
const FLOAT chip_width = 0.016;

/* ambient temperature, assuming no package at all	*/
const FLOAT amb_temp = 80.0;

/* stencil parameters */
const uindex_t stencil_radius = 1;
const uindex_t pipeline_length = 256;
const uindex_t tile_width = 1024;
const uindex_t tile_height = 1024;
const uindex_t burst_size = 1024;

#ifdef HARDWARE
using selector = INTEL::fpga_selector;
#else
using selector = INTEL::fpga_emulator_selector;
#endif

void write_output(buffer<vec<FLOAT, 2>, 2> vect, string file) {
    fstream out(file, out.out | out.trunc);
    if (!out.is_open()) {
        throw std::runtime_error("The file was not opened\n");
    }

    uindex_t n_columns = vect.get_range()[0];
    uindex_t n_rows = vect.get_range()[1];
    auto vect_ac = vect.get_access<access::mode::read>();

    int i = 0;
    for (index_t r = 0; r < n_rows; r++) {
        for (index_t c = 0; c < n_columns; c++) {
            out << i << "\t" << vect_ac[id<2>(c, r)][0] << std::endl;
            i++;
        }
    }

    out.close();
}

buffer<vec<FLOAT, 2>, 2> read_input(string temp_file, string power_file, range<2> buffer_range) {
    fstream temp(temp_file, temp.in);
    fstream power(power_file, power.in);
    if (!temp.is_open() || !power.is_open()) {
        throw std::runtime_error("file could not be opened for reading");
    }

    uindex_t n_columns = buffer_range[0];
    uindex_t n_rows = buffer_range[1];
    buffer<vec<FLOAT, 2>, 2> vect(buffer_range);

    {
        auto vect_ac = vect.get_access<access::mode::write>();

        FLOAT tmp_temp, tmp_power;
        for (index_t r = 0; r < n_rows; r++) {
            for (index_t c = 0; c < n_columns; c++) {
                temp >> tmp_temp;
                power >> tmp_power;
                vect_ac[id<2>(c, r)] = vec<FLOAT, 2>(tmp_temp, tmp_power);
            }
        }
    }

    temp.close();
    power.close();
    return vect;
}

void usage(int argc, char **argv) {
    std::cerr << "Usage: " << argv[0]
              << "  <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>"
              << std::endl;
    std::cerr << "    <grid_rows>      - number of rows in the grid (positive integer)"
              << std::endl;
    std::cerr << "    <grid_cols>      - number of columns in the grid (positive integer)"
              << std::endl;
    std::cerr << "    <sim_time>       - number of iterations (positive integer)" << std::endl;
    std::cerr << "    <temp_file>      - name of the file containing the initial temperature "
                 "values of each cell"
              << std::endl;
    std::cerr << "    <power_file>     - name of the file containing the dissipated power values "
                 "of each cell"
              << std::endl;
    std::cerr << "    <output_file>    - name of the output file" << std::endl;
    exit(1);
}

auto exception_handler = [](cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (cl::sycl::exception const &e) {
            std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
            std::terminate();
        }
    }
};

double run_simulation(cl::sycl::queue working_queue, buffer<vec<FLOAT, 2>, 2> temp,
                      uindex_t sim_time) {
    uindex_t n_columns = temp.get_range()[0];
    uindex_t n_rows = temp.get_range()[1];

    FLOAT grid_height = chip_height / n_rows;
    FLOAT grid_width = chip_width / n_columns;

    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1 = 1.f / Rx;
    FLOAT Ry_1 = 1.f / Ry;
    FLOAT Rz_1 = 1.f / Rz;
    FLOAT Cap_1 = step / Cap;

    auto kernel = [=](Stencil<vec<FLOAT, 2>, stencil_radius> const &temp) {
        ID idx = temp.id;
        index_t c = idx.c;
        index_t r = idx.r;
        uindex_t width = temp.grid_range.c;
        uindex_t height = temp.grid_range.r;

        FLOAT power = temp[ID(0, 0)][1];
        FLOAT old = temp[ID(0, 0)][0];
        FLOAT left = temp[ID(-1, 0)][0];
        FLOAT right = temp[ID(1, 0)][0];
        FLOAT top = temp[ID(0, -1)][0];
        FLOAT bottom = temp[ID(0, 1)][0];

        if (c == 0) {
            left = old;
        } else if (c == width - 1) {
            right = old;
        }

        if (r == 0) {
            top = old;
        } else if (r == height - 1) {
            bottom = old;
        }

        // As in the OpenCL version of the rodinia "hotspot" benchmark.
        FLOAT new_temp =
            old + Cap_1 * (power + (bottom + top - 2.f * old) * Ry_1 +
                           (right + left - 2.f * old) * Rx_1 + (amb_temp - old) * Rz_1);

        return vec(new_temp, power);
    };

#ifdef MONOTILE
    using Executor = MonotileExecutor<vec<FLOAT, 2>, stencil_radius, decltype(kernel),
                                      pipeline_length, tile_width, tile_height>;
#else
    using Executor = StencilExecutor<vec<FLOAT, 2>, stencil_radius, decltype(kernel),
                                     pipeline_length, tile_width, tile_height, burst_size>;
#endif

    Executor executor(vec<FLOAT, 2>(0.0, 0.0), kernel);
    executor.set_input(temp);

#ifdef HARDWARE
    executor.select_fpga(true);
#else
    executor.select_emulator(true);
#endif

    executor.run(sim_time);

    executor.copy_output(temp);

    return executor.get_runtime_sample().value().get_total_runtime();
}

int main(int argc, char **argv) {
    int n_rows, n_columns, sim_time;
    char *tfile, *pfile, *ofile;
    bool benchmark_mode = false;

    selector device_selector;
    cl::sycl::queue working_queue(device_selector, exception_handler,
                                  {property::queue::enable_profiling{}});

    /* check validity of inputs	*/
    if (argc != 7)
        usage(argc, argv);
    if ((n_rows = atoi(argv[1])) <= 0)
        usage(argc, argv);
    if ((n_columns = atoi(argv[2])) <= 0)
        usage(argc, argv);
    if ((sim_time = atoi(argv[3])) <= 0)
        usage(argc, argv);

#ifdef MONOTILE
    if (n_columns > tile_width || n_rows > tile_height) {
        std::cerr << "Error: The grid may not exceed the size of the tile (" << tile_width << " by "
                  << tile_height << " cells) when using the monotile architecture." << std::endl;
        exit(1);
    }
#endif

    /* read initial temperatures and input power	*/
    tfile = argv[4];
    pfile = argv[5];
    ofile = argv[6];
    buffer<vec<FLOAT, 2>, 2> temp =
        read_input(string(tfile), string(pfile), range<2>(n_columns, n_rows));

    printf("Start computing the transient temperature\n");

    double runtime = run_simulation(working_queue, temp, sim_time);

    printf("Ending simulation\n");
    std::cout << "Total time: " << runtime << " s" << std::endl;
    write_output(temp, string(ofile));

    return 0;
}
