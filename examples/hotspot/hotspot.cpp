/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include <StencilStream/BaseTransitionFunction.hpp>
#include <chrono>
#include <fstream>
#include <sycl/ext/intel/fpga_extensions.hpp>

#if defined(STENCILSTREAM_BACKEND_MONOTILE)
    #include <StencilStream/monotile/StencilUpdate.hpp>
#elif defined(STENCILSTREAM_BACKEND_TILING)
    #include <StencilStream/tiling/StencilUpdate.hpp>
#elif defined(STENCILSTREAM_BACKEND_CPU)
    #include <StencilStream/cpu/StencilUpdate.hpp>
#endif

using namespace std;
using namespace sycl;
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
using HotspotCell = vec<FLOAT, 2>;

struct HotspotKernel : public BaseTransitionFunction {
    using Cell = HotspotCell;

    float Rx_1, Ry_1, Rz_1, Cap_1;

    Cell operator()(Stencil<HotspotCell, 1> const &temp) const {
        using StencilID = typename Stencil<HotspotCell, 1>::StencilID;

        ID idx = temp.id;
        index_t c = idx.c;
        index_t r = idx.r;
        uindex_t width = temp.grid_range.c;
        uindex_t height = temp.grid_range.r;

        FLOAT power = temp[StencilID(0, 0)][1];
        FLOAT old = temp[StencilID(0, 0)][0];
        FLOAT left = temp[StencilID(-1, 0)][0];
        FLOAT right = temp[StencilID(1, 0)][0];
        FLOAT top = temp[StencilID(0, -1)][0];
        FLOAT bottom = temp[StencilID(0, 1)][0];

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
    }
};

#if defined(STENCILSTREAM_BACKEND_MONOTILE)
const uindex_t max_grid_width = 1024;
const uindex_t max_grid_height = 1024;
const uindex_t n_processing_elements = 280;
using StencilUpdate =
    monotile::StencilUpdate<HotspotKernel, n_processing_elements, max_grid_width, max_grid_height>;
using Grid = monotile::Grid<HotspotCell>;

#elif defined(STENCILSTREAM_BACKEND_TILING)
const uindex_t tile_width = 1 << 16;
const uindex_t tile_height = 1024;
const uindex_t n_processing_elements = 224;
using StencilUpdate =
    tiling::StencilUpdate<HotspotKernel, n_processing_elements, tile_width, tile_height>;
using Grid = StencilUpdate::GridImpl;

#elif defined(STENCILSTREAM_BACKEND_CPU)
using StencilUpdate = cpu::StencilUpdate<HotspotKernel>;
using Grid = StencilUpdate::GridImpl;

#endif

void write_output(Grid vect, string file, bool binary) {
    fstream out;
    if (binary) {
        out = fstream(file, out.out | out.trunc | out.binary);
    } else {
        out = fstream(file, out.out | out.trunc);
    }

    if (!out.is_open()) {
        throw std::runtime_error("The file was not opened\n");
    }

    uindex_t n_columns = vect.get_grid_width();
    uindex_t n_rows = vect.get_grid_height();
    Grid::GridAccessor<access::mode::read> vect_ac(vect);

    int i = 0;
    for (index_t r = 0; r < n_rows; r++) {
        for (index_t c = 0; c < n_columns; c++) {
            if (binary) {
                out.write((char *)&vect_ac[c][r][0], sizeof(float));
            } else {
                out << i << "\t" << vect_ac[c][r][0] << std::endl;
            }
            i++;
        }
    }

    out.close();
}

Grid read_input(string temp_file, string power_file, uindex_t n_columns, uindex_t n_rows,
                bool binary) {
    fstream temp, power;
    if (binary) {
        temp = fstream(temp_file, temp.in | temp.binary);
        power = fstream(power_file, power.in | power.binary);
    } else {
        temp = fstream(temp_file, temp.in);
        power = fstream(power_file, power.in);
    }

    Grid vect(n_columns, n_rows);
    {
        Grid::GridAccessor<access::mode::read_write> vect_ac(vect);

        for (index_t r = 0; r < n_rows; r++) {
            for (index_t c = 0; c < n_columns; c++) {
                FLOAT tmp_temp, tmp_power;
                if (binary) {
                    temp.read((char *)&tmp_temp, sizeof(float));
                    power.read((char *)&tmp_power, sizeof(float));
                } else {
                    temp >> tmp_temp;
                    power >> tmp_power;
                }
                vect_ac[c][r] = HotspotCell(tmp_temp, tmp_power);
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

int main(int argc, char **argv) {
    uindex_t n_rows, n_columns, sim_time;
    bool benchmark_mode = false;

    /* check validity of inputs	*/
    if (argc != 7)
        usage(argc, argv);
    if ((n_rows = atoi(argv[1])) <= 0)
        usage(argc, argv);
    if ((n_columns = atoi(argv[2])) <= 0)
        usage(argc, argv);
    if ((sim_time = atoi(argv[3])) <= 0)
        usage(argc, argv);

#if defined(STENCILSTREAM_BACKEND_MONOTILE)
    if (n_columns > max_grid_width || n_rows > max_grid_height) {
        std::cerr << "Error: The grid may not exceed a size of " << max_grid_width << " by "
                  << max_grid_height << " cells when using the monotile architecture." << std::endl;
        exit(1);
    }
#endif

    /* read initial temperatures and input power	*/
    std::string tfile = std::string(argv[4]);
    std::string pfile = std::string(argv[5]);
    std::string ofile = std::string(argv[6]);

    bool binary_io = tfile.ends_with(".bin");
    assert(!binary_io || pfile.ends_with(".bin"));

    Grid grid = read_input(tfile, pfile, n_columns, n_rows, binary_io);

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

#if defined(STENCILSTREAM_TARGET_FPGA)
    sycl::device device(sycl::ext::intel::fpga_selector_v);
#else
    sycl::device device;
#endif

    StencilUpdate update({
        .transition_function =
            HotspotKernel{.Rx_1 = Rx_1, .Ry_1 = Ry_1, .Rz_1 = Rz_1, .Cap_1 = Cap_1},
        .halo_value = HotspotCell(0.0, 0.0), .n_iterations = sim_time, .device = device,
        .blocking = true, // enable blocking for meaningful walltime measurements
#if !defined(STENCILSTREAM_BACKEND_CPU)
            .profiling = true, // enable additional profiling for FPGA targets
#endif
    });

    grid = update(grid);

    std::cout << "Ending simulation" << std::endl;
    std::cout << "Walltime: " << update.get_walltime() << " s" << std::endl;
#if !defined(STENCILSTREAM_BACKEND_CPU)
    // Print pure kernel runtime for FPGA targets
    std::cout << "Kernel Runtime: " << update.get_kernel_runtime() << " s" << std::endl;
#endif

    write_output(grid, ofile, binary_io);

    return 0;
}
