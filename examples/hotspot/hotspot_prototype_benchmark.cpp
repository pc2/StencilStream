#include "../../StencilStream/cuda/StencilUpdate_hotspot_prototype.hpp"
#include <benchmark/benchmark.h>
#include <chrono>
#include <fstream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

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
using HotspotCell = vec<FLOAT, 2>;

struct HotspotKernel {
    using Cell = HotspotCell;
    static constexpr std::size_t stencil_radius = 1;
    static constexpr std::size_t n_subiterations = 1;
    float Rx_1, Ry_1, Rz_1, Cap_1;

    Cell operator()(Stencil<HotspotCell, 1> const &temp) const {
        FLOAT power = temp[0][0][1];
        FLOAT old = temp[0][0][0];
        FLOAT top = temp[-1][0][0];
        FLOAT bottom = temp[1][0][0];
        FLOAT left = temp[0][-1][0];
        FLOAT right = temp[0][1][0];

        if (temp.id[0] == 0) {
            top = old;
        } else if (temp.id[0] == temp.grid_range[0] - 1) {
            bottom = old;
        }

        if (temp.id[1] == 0) {
            left = old;
        } else if (temp.id[1] == temp.grid_range[1] - 1) {
            right = old;
        }

        // As in the OpenCL version of the rodinia "hotspot" benchmark.
        FLOAT new_temp =
            old + Cap_1 * (power + (bottom + top - 2.f * old) * Ry_1 +
                           (right + left - 2.f * old) * Rx_1 + (amb_temp - old) * Rz_1);

        return vec(new_temp, power);
    }
};

void write_output(Grid<HotspotCell> vect, string file, bool binary) {
    fstream out;
    if (binary) {
        out = fstream(file, out.out | out.trunc | out.binary);
    } else {
        out = fstream(file, out.out | out.trunc);
    }

    if (!out.is_open()) {
        throw std::runtime_error("The file was not opened\n");
    }

    Grid<HotspotCell>::GridAccessor<access::mode::read> vect_ac(vect);

    int i = 0;
    for (size_t r = 0; r < vect.get_grid_height(); r++) {
        for (size_t c = 0; c < vect.get_grid_width(); c++) {
            if (binary) {
                out.write((char *)&vect_ac[r][c][0], sizeof(float));
            } else {
                out << i << "\t" << vect_ac[r][c][0] << std::endl;
            }
            i++;
        }
    }

    out.close();
}

Grid<HotspotCell> read_input(string temp_file, string power_file, size_t n_rows, size_t n_columns,
                             bool binary) {
    fstream temp, power;
    if (binary) {
        temp = fstream(temp_file, temp.in | temp.binary);
        power = fstream(power_file, power.in | power.binary);
    } else {
        temp = fstream(temp_file, temp.in);
        power = fstream(power_file, power.in);
    }

    Grid<HotspotCell> vect(n_rows, n_columns);
    {
        Grid<HotspotCell>::GridAccessor<access::mode::read_write> vect_ac(vect);

        for (size_t r = 0; r < n_rows; r++) {
            for (size_t c = 0; c < n_columns; c++) {
                FLOAT tmp_temp, tmp_power;
                if (binary) {
                    temp.read((char *)&tmp_temp, sizeof(float));
                    power.read((char *)&tmp_power, sizeof(float));
                } else {
                    temp >> tmp_temp;
                    power >> tmp_power;
                }
                vect_ac[r][c] = HotspotCell(tmp_temp, tmp_power);
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

auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const &e) {
            std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << "\n";
            std::terminate();
        }
    }
};

static void BM_HotspotKernel(benchmark::State &state) {
    // Args: 0=row, 1=col, 2=sim_time, 3=temp_file_index, 4=power_file_index
    size_t n_rows = state.range(0);
    size_t n_columns = state.range(1);
    size_t sim_time = state.range(2);
    std::string temp_file = "temp_" + std::to_string(state.range(3));
    std::string power_file = "power_" + std::to_string(state.range(4));
    std::string output_file = "/dev/null"; // Kein Output nötig beim Benchmark

    bool binary_io = false;
    Grid<HotspotCell> grid = read_input(temp_file, power_file, n_rows, n_columns, binary_io);

    FLOAT grid_height = chip_height / n_rows;
    FLOAT grid_width = chip_width / n_columns;

    FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_height * grid_width;
    FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

    FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    FLOAT step = PRECISION / max_slope / 1000.0;

    FLOAT Rx_1 = 1.f / Rx;
    FLOAT Ry_1 = 1.f / Ry;
    FLOAT Rz_1 = 1.f / Rz;
    FLOAT Cap_1 = step / Cap;

    sycl::device device(sycl::gpu_selector_v);

    StencilUpdate<HotspotKernel> update({
        .transition_function =
            HotspotKernel{.Rx_1 = Rx_1, .Ry_1 = Ry_1, .Rz_1 = Rz_1, .Cap_1 = Cap_1},
        .halo_value = HotspotCell(0.0, 0.0),
        .n_iterations = sim_time,
        .device = device,
        .blocking = true,
        .profiling = true,
    });

    for (auto _ : state) {
        Grid<HotspotCell> local_grid = grid;
        local_grid = update(local_grid);
    }
    /*
    double data_prep_time_s =
        (update.get_data_preperation_time_before() + update.get_data_preperation_time_after()) /
            state.iterations() / 1.0e9; */

    double kernel_time = update.get_kernel_runtime();
    double avg_kernel_time = kernel_time / state.iterations();
    double wall_time = update.get_walltime();
    double avg_wall_time = wall_time / state.iterations();
    double avg_data_prep_time =
        (update.get_data_preperation_time_before() + update.get_data_preperation_time_after()) /
        state.iterations();

    double flops = state.iterations() * 1.0 * n_rows * n_columns * 15.0 * sim_time;
    double factor = avg_wall_time / avg_kernel_time; // >1 = Overhead

    state.counters["Avg_Kernel_time"] = avg_kernel_time;
    state.counters["Avg_Wall_time"] = avg_wall_time;
    state.counters["Sim_time"] = benchmark::Counter(sim_time, benchmark::Counter::kDefaults);
    state.counters["Flops_walltime"] = flops / wall_time;
    state.counters["Flops_kerneltime"] = flops / kernel_time;
    state.counters["Kerneltime to walltime"] = factor;
    state.counters["Data_prep_time"] = avg_data_prep_time;
}

void CustomArgs(benchmark::internal::Benchmark *b) {

    int iter = 28;
    while (iter > 18) {
        b->Args({1024, 1024, iter, 1024, 1024});
        iter -= 1;
    }
}

BENCHMARK(BM_HotspotKernel)->Apply(CustomArgs)->Unit(benchmark::kSecond);

BENCHMARK_MAIN();
