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
#include <fstream>
#include <stencil/stencil.hpp>

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
const stencil::uindex_t stencil_radius = 1;
const stencil::uindex_t max_width = 1024;
const stencil::uindex_t max_height = 1024;

/* Number of simulations to run in benchmark mode */
const uindex_t n_simulations = 10;

void write_output(buffer<vec<FLOAT, 2>, 2> vect, string file)
{
    fstream out(file, out.out | out.trunc);
    if (!out.is_open())
    {
        throw std::runtime_error("The file was not opened\n");
    }

    uindex_t n_columns = vect.get_range()[0];
    uindex_t n_rows = vect.get_range()[1];
    auto vect_ac = vect.get_access<access::mode::read>();

    int i = 0;
    for (index_t r = 0; r < n_rows; r++)
    {
        for (index_t c = 0; c < n_columns; c++)
        {
            out << i << "\t" << vect_ac[id<2>(c, r)][0] << std::endl;
            i++;
        }
    }

    out.close();
}

buffer<vec<FLOAT, 2>, 2>
read_input(string temp_file, string power_file, range<2> buffer_range)
{
    fstream temp(temp_file, temp.in);
    fstream power(power_file, power.in);
    if (!temp.is_open() || !power.is_open())
    {
        throw std::runtime_error("file could not be opened for reading");
    }

    uindex_t n_columns = buffer_range[0];
    uindex_t n_rows = buffer_range[1];
    buffer<vec<FLOAT, 2>, 2> vect(buffer_range);

    {
        auto vect_ac = vect.get_access<access::mode::write>();

        FLOAT tmp_temp, tmp_power;
        for (index_t r = 0; r < n_rows; r++)
        {
            for (index_t c = 0; c < n_columns; c++)
            {
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

void usage(int argc, char **argv)
{
    std::cerr << "Usage: " << argv[0] << "  <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file> [<benchmark_mode>]" << std::endl;
    std::cerr << "    <grid_rows>      - number of rows in the grid (positive integer)" << std::endl;
    std::cerr << "    <grid_cols>      - number of columns in the grid (positive integer)" << std::endl;
    std::cerr << "    <sim_time>       - number of iterations (positive integer)" << std::endl;
    std::cerr << "    <temp_file>      - name of the file containing the initial temperature values of each cell" << std::endl;
    std::cerr << "    <power_file>     - name of the file containing the dissipated power values of each cell" << std::endl;
    std::cerr << "    <output_file>    - name of the output file" << std::endl;
    std::cerr << "    <benchmark_mode> - Either 'true' or 'false', default 'false'. Run simulation multiple times and analyze the performance" << std::endl;
    std::cerr << "                       " << n_simulations << " simulations with i*<sim_time> generations will be executed in total, where i is the index of the simulation." << std::endl;
    std::cerr << "This build supports grids with up to " << max_height << " rows." << std::endl;
    exit(1);
}

auto exception_handler = [](cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions)
    {
        try
        {
            std::rethrow_exception(e);
        }
        catch (cl::sycl::exception const &e)
        {
            std::cout << "Caught asynchronous SYCL exception:\n"
                      << e.what() << "\n";
            std::terminate();
        }
    }
};

event run_simulation(cl::sycl::queue working_queue, buffer<vec<FLOAT, 2>, 2> temp, uindex_t sim_time)
{
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

    auto kernel =
        [=](
            stencil::Stencil2D<vec<FLOAT, 2>, stencil_radius> const &temp,
            stencil::Stencil2DInfo const &parameter) {
            auto idx = parameter.center_cell_id;

            FLOAT power = temp[ID(0, 0)][1];
            FLOAT old = temp[ID(0, 0)][0];
            FLOAT left = temp[ID(-1, 0)][0];
            FLOAT right = temp[ID(1, 0)][0];
            FLOAT top = temp[ID(0, -1)][0];
            FLOAT bottom = temp[ID(0, 1)][0];

            index_t c = idx.c;
            index_t r = idx.r;

            FLOAT horizontal_sum;
            if (c == 0)
            {
                horizontal_sum = right - old;
            }
            else if (c == n_columns - 1)
            {
                horizontal_sum = left - old;
            }
            else
            {
                horizontal_sum = right + left - 2.f * old;
            }
            horizontal_sum *= Rx_1;

            FLOAT vertical_sum;
            bool full_vertical = false;
            if (r == 0)
            {
                vertical_sum = bottom - old;
            }
            else if (r == n_rows - 1)
            {
                vertical_sum = top - old;
            }
            else
            {
                vertical_sum = bottom + top - 2.f * old;
                full_vertical = true;
            }
            vertical_sum *= Ry_1;

            FLOAT ambient = (amb_temp - old) * Rz_1;

            FLOAT sum;
            if (full_vertical)
            {
                sum = power + vertical_sum + horizontal_sum + ambient;
            }
            else
            {
                sum = power + horizontal_sum + vertical_sum + ambient;
            }

            return vec(old + Cap_1 * sum, power);
        };

    StencilExecutor<vec<FLOAT, 2>, stencil_radius, max_width, max_height> executor(working_queue);
    executor.set_buffer(temp);
    executor.set_generations(sim_time);
    return executor.run(kernel);
}

int main(int argc, char **argv)
{
    int n_rows, n_columns, sim_time;
    char *tfile, *pfile, *ofile;
    bool benchmark_mode = false;

#ifdef HARDWARE
    INTEL::fpga_selector device_selector;
#else
    INTEL::fpga_emulator_selector device_selector;
#endif
    cl::sycl::queue working_queue(device_selector, exception_handler, {property::queue::enable_profiling{}});

    /* check validity of inputs	*/
    if (argc < 7 || argc > 8)
        usage(argc, argv);
    if ((n_rows = atoi(argv[1])) <= 0)
        usage(argc, argv);
    if ((n_columns = atoi(argv[2])) <= 0)
        usage(argc, argv);
    if ((sim_time = atoi(argv[3])) <= 0)
        usage(argc, argv);

    /* read initial temperatures and input power	*/
    tfile = argv[4];
    pfile = argv[5];
    ofile = argv[6];
    buffer<vec<FLOAT, 2>, 2> temp = read_input(string(tfile), string(pfile), range<2>(n_columns, n_rows));
    if (argc == 8 && std::string(argv[7]) == std::string("true"))
    {
        benchmark_mode = true;
    }

    if (benchmark_mode)
    {
        std::cout << "Starting benchmark" << std::endl;

        double clock_frequency;
        std::cout << "Clock frequency: ";
        std::cin >> clock_frequency;
        std::cout << std::endl;

        double runtimes[n_simulations];
        for (uindex_t i = 0; i < n_simulations; i++)
        {
            event comp_event = run_simulation(working_queue, temp, (i + 1) * sim_time);

            unsigned long event_start = comp_event.get_profiling_info<info::event_profiling::command_start>();
            unsigned long event_end = comp_event.get_profiling_info<info::event_profiling::command_end>();
            runtimes[i] = double(event_end - event_start) / 1000000000.0;
            std::cout << "Run " << i << " with " << (i + 1) * sim_time << " passes took " << runtimes[i] << " seconds" << std::endl;
        }

        // Actually the mean delta seconds per pass.
        double delta_seconds_per_pass = 0.0;
        for (uindex_t i = 0; i < n_simulations - 1; i++)
        {
            delta_seconds_per_pass += abs(double(runtimes[i + 1]) - double(runtimes[i]));
        }
        delta_seconds_per_pass = (delta_seconds_per_pass / (n_simulations - 1)) * (pipeline_length / sim_time);

        std::cout << "Time per buffer pass: " << delta_seconds_per_pass << "s" << std::endl;

        double loops_per_pass = max_width * max_height;
        double seconds_per_loop = delta_seconds_per_pass / loops_per_pass;
        double cycles_per_loop = seconds_per_loop * clock_frequency;

        std::cout << "Cycles per Loop, aka II.: " << cycles_per_loop << std::endl;

        double fo_per_core = 15;
        double fo_per_loop = fo_per_core * pipeline_length;
        double fo_per_pass = fo_per_loop * loops_per_pass;
        double fo_per_second = fo_per_pass / delta_seconds_per_pass;

        std::cout << "Raw Performance: " << fo_per_second / 1000000000.0 << " GFLOPS" << std::endl;

        double generations_per_pass = pipeline_length;
        double generations_per_second = generations_per_pass / delta_seconds_per_pass;

        std::cout << "Performance: " << generations_per_second << " Generations/s" << std::endl;
    }
    else
    {
        printf("Start computing the transient temperature\n");

        event comp_event = run_simulation(working_queue, temp, sim_time);

        printf("Ending simulation\n");

        unsigned long event_start = comp_event.get_profiling_info<info::event_profiling::command_start>();
        unsigned long event_end = comp_event.get_profiling_info<info::event_profiling::command_end>();
        std::cout << "Total time: " << (event_end - event_start) / 1000000000.0 << " s" << std::endl;

        write_output(temp, string(ofile));
    }

    return 0;
}
