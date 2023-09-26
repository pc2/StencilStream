/*
 * Copyright © 2020-2023 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include <chrono>
#include <fstream>
#include <sycl/ext/intel/fpga_extensions.hpp>

#if EXECUTOR == 0
    #include <StencilStream/monotile/StencilUpdate.hpp>
#elif EXECUTOR == 1
    #include <StencilStream/tiling/StencilUpdate.hpp>
#elif EXECUTOR == 2
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

struct HotspotKernel {
    using Cell = HotspotCell;
    using TimeDependentValue = std::monostate;

    static constexpr uindex_t stencil_radius = 1;
    static constexpr uindex_t n_subgenerations = 1;

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

#if EXECUTOR == 0
const uindex_t max_grid_height = 1024;
const uindex_t n_processing_elements = 350;
using StencilUpdate = monotile::StencilUpdate<HotspotKernel, tdv::NoneSupplier,
                                              n_processing_elements, max_grid_height>;
using Grid = monotile::Grid<HotspotCell>;

#elif EXECUTOR == 1
const uindex_t tile_width = 1024;
const uindex_t tile_height = 1024;
const uindex_t n_processing_elements = 280;
using StencilUpdate = tiling::StencilUpdate<HotspotKernel, tdv::NoneSupplier, n_processing_elements,
                                            tile_width, tile_height>;
using Grid = StencilUpdate::GridImpl;

#else
using StencilUpdate = cpu::StencilUpdate<HotspotKernel>;
using Grid = StencilUpdate::GridImpl;

#endif

void write_output(Grid vect, string file) {
    fstream out(file, out.out | out.trunc);
    if (!out.is_open()) {
        throw std::runtime_error("The file was not opened\n");
    }

    uindex_t n_columns = vect.get_grid_width();
    uindex_t n_rows = vect.get_grid_height();
    Grid::GridAccessor<access::mode::read> vect_ac(vect);

    int i = 0;
    for (index_t r = 0; r < n_rows; r++) {
        for (index_t c = 0; c < n_columns; c++) {
            out << i << "\t" << vect_ac.get(c, r)[0] << std::endl;
            i++;
        }
    }

    out.close();
}

Grid read_input(string temp_file, string power_file, uindex_t n_columns, uindex_t n_rows) {
    fstream temp(temp_file, temp.in);
    fstream power(power_file, power.in);
    if (!temp.is_open() || !power.is_open()) {
        throw std::runtime_error("file could not be opened for reading");
    }

    Grid vect(n_columns, n_rows);
    {
        Grid::GridAccessor<access::mode::read_write> vect_ac(vect);

        for (index_t r = 0; r < n_rows; r++) {
            for (index_t c = 0; c < n_columns; c++) {
                FLOAT tmp_temp, tmp_power;
                temp >> tmp_temp;
                power >> tmp_power;
                vect_ac.set(c, r, HotspotCell(tmp_temp, tmp_power));
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
    char *tfile, *pfile, *ofile;
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

#if EXECUTOR == 0
    if (n_rows > max_grid_height) {
        std::cerr << "Error: The grid may not exceed a height of " << max_grid_height
                  << " cells when using the monotile architecture." << std::endl;
        exit(1);
    }
#endif

    /* read initial temperatures and input power	*/
    tfile = argv[4];
    pfile = argv[5];
    ofile = argv[6];
    Grid grid = read_input(string(tfile), string(pfile), n_columns, n_rows);

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

#if HARDWARE == 1
    sycl::queue queue(sycl::ext::intel::fpga_selector_v);
#else
    sycl::queue queue;
#endif

    StencilUpdate update({
        .transition_function = HotspotKernel{Rx_1, Ry_1, Rz_1, Cap_1},
        .halo_value = HotspotCell(0.0, 0.0),
        .n_generations = sim_time,
        .queue = queue,
    });

    auto start = std::chrono::high_resolution_clock::now();
    grid = update(grid);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end - start;

    printf("Ending simulation\n");
    std::cout << "Total time: " << runtime.count() << " s" << std::endl;
    write_output(grid, string(ofile));

    return 0;
}
