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
            out << i << "\t" << vect_ac[id<2>(r, c)][0] << std::endl;
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
                vect_ac[id<2>(r, c)] = vec<FLOAT, 2>(tmp_temp, tmp_power);
            }
        }
    }

    temp.close();
    power.close();
    return vect;
}

void usage(int argc, char **argv)
{
    std::cerr << "Usage: " << argv[0] << "  <sim_time> <temp_file> <power_file>" << std::endl;
    std::cerr << "    <grid_rows>   - number of rows in the grid (positive integer)" << std::endl;
    std::cerr << "    <grid_cols>   - number of columns in the grid (positive integer)" << std::endl;
    std::cerr << "    <sim_time>    - number of iterations (positive integer)" << std::endl;
    std::cerr << "    <temp_file>   - name of the file containing the initial temperature values of each cell" << std::endl;
    std::cerr << "    <power_file>  - name of the file containing the dissipated power values of each cell" << std::endl;
    std::cerr << "    <output_file> - name of the output file" << std::endl;
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

int main(int argc, char **argv)
{
    int n_rows, n_columns, sim_time;
    char *tfile, *pfile, *ofile;

#ifdef HARDWARE
    INTEL::fpga_selector device_selector;
#else
    INTEL::fpga_emulator_selector device_selector;
#endif
    cl::sycl::queue working_queue(device_selector, exception_handler, {property::queue::enable_profiling{}});

    /* check validity of inputs	*/
    if (argc != 7)
        usage(argc, argv);
    if ((n_rows = atoi(argv[1])) <= 0)
        usage(argc, argv);
    if ((n_columns = atoi(argv[2])) <= 0)
        usage(argc, argv);
    if ((sim_time = atoi(argv[3])) < 0)
        usage(argc, argv);

    /* read initial temperatures and input power	*/
    tfile = argv[4];
    pfile = argv[5];
    ofile = argv[6];
    buffer<vec<FLOAT, 2>, 2> temp = read_input(string(tfile), string(pfile), range<2>(n_columns, n_rows));

    printf("Start computing the transient temperature\n");

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
    event comp_event = executor.run(kernel);

    printf("Ending simulation\n");

    unsigned long event_start = comp_event.get_profiling_info<info::event_profiling::command_start>();
    unsigned long event_end = comp_event.get_profiling_info<info::event_profiling::command_end>();
    std::cout << "Total time: " << (event_end - event_start) / 1000000000.0 << " s" << std::endl;

    write_output(temp, string(ofile));

    return 0;
}
