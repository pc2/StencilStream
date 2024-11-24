#include <StencilStream/BaseTransitionFunction.hpp>
#include <StencilStream/cpu/StencilUpdate.hpp>
#include <StencilStream/monotile/StencilUpdate.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "CL/sycl.hpp"

#include <chrono>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <vector>
#include <string>

using wall_clock_t = std::chrono::high_resolution_clock;
using namespace stencil;
#if defined(STENCILSTREAM_BACKEND_CPU)
using namespace stencil::cpu;
#else
using namespace stencil::monotile;
#endif

struct field
{
    int nx, ny;
    double dx, dy;
    std::vector<double> data;
};

constexpr auto DX = 0.01;
constexpr auto DY = 0.01;

struct HeatKernel : public BaseTransitionFunction
{
    using Cell = double;

    double a, dt;
    uindex_t rows, cols;

    HeatKernel(double a_, double dt_, uindex_t rows_, uindex_t cols_) : a(a_), dt(dt_), rows(rows_), cols(cols_) {}

    double operator()(Stencil<double, 1> const &stencil) const
    {
        auto [c, r] = stencil.id;

        // Boundary conditions
        if (r == 0) {
            // Top boundary
            return 85.0;
        } else if (r == rows - 1) {
            // Bottom boundary
            return 5.0;
        } else if (c == 0) {
            // Left boundary
            return 20.0;
        } else if (c == cols - 1) {
            // Right boundary
            return 70.0;
        } else {
            // Stencil design
            double center = stencil[ID(0, 0)];
            double left = stencil[ID(-1, 0)];
            double right = stencil[ID(1, 0)];
            double top = stencil[ID(0, -1)];
            double bottom = stencil[ID(0, 1)];

            // Calculation
            return center + a * dt * ((left - 2.0 * center + right) / (DX * DX) +
                                      (top - 2.0 * center + bottom) / (DY * DY));
        }
    }
};

// Generation of the field with the boundary conditions
void initialize(field *field, uindex_t rows, uindex_t cols)
{
    field->dx = DX;
    field->dy = DY;
    field->nx = rows;
    field->ny = cols;

    int ind;
    double radius;
    int dx, dy;

    int newSize = field->nx * field->ny;
    field->data.resize(newSize, 0.0);

    radius = field->nx / 6.0;
    for (int i = 0; i < field->nx ; i++)
    {
        for (int j = 0; j < field->ny ; j++)
        {
            ind = i * field->ny + j;
            
            dx = i - field->nx / 2;
            dy = j - field->ny / 2;
            if (dx * dx + dy * dy < radius * radius)
            {
                field->data[ind] = 5.0;
            }
            else
            {
                field->data[ind] = 65.0;
            }
        }
    }

    // Boundary conditions
    for (int i = 0; i < field->nx ; i++)
    {
        field->data[i * field->ny + field->ny - 1] = 70.0;
        field->data[i * field->ny ] = 20.0;
    }

    for (int j = 0; j < field->ny ; j++)
    {
        field->data[j] = 85.0;
    }
    for (int j = 0; j < field->ny ; j++)
    {
        field->data[(field->nx - 1)* field->ny + j] = 5.0;
    }
}

//void print_field(field *field)
//{
    //int stride = field->ny;

    //for (int i = 0; i < field->nx; i++)
    //{
        //for (int j = 0; j < field->ny; j++) 
        //{
            //std::cout << field->data[(i * stride) + j] << " ";
        //}
        //std::cout << std::endl;
    //}
//}

void write(Grid<double> output_grid) {
    // TODO: Change output naming
    // TODO: Add a loop for naming 
    std::string name = "Output.csv";
    std::ofstream myFile(name);

    Grid<double>::GridAccessor<sycl::access::mode::read> grid_ac(output_grid);

    for (uindex_t r = 0; r < output_grid.get_grid_height(); r++) {
        for (uindex_t c = 0; c < output_grid.get_grid_width(); c++) {
            //std::cout << grid_ac[c][r] << ' ';
            myFile << grid_ac[c][r];
            if(c + 1 != output_grid.get_grid_width()) {
                myFile << ",";
            }
        }
        //std::cout << std::endl;
        myFile << std::endl;
    }
}

Grid<double> read(field *field, uindex_t rows, uindex_t cols) {
    Grid<double> input_grid(rows, cols);
    {
        Grid<double>::GridAccessor<sycl::access::mode::read_write> grid_ac(input_grid);

        uindex_t stride = input_grid.get_grid_height(); 
        for (uindex_t r = 0; r < cols; r++) {
            for (uindex_t c = 0; c < rows; c++) {
                grid_ac[c][r] = field->data[r * stride + c];
            }
        }
    }
    return input_grid;
}

//void printField(field *field){
    //int stride = field->ny; // Total number of columns, including ghost layers

    //for (int i = 0; i < field->nx; i++) // Loop over all rows, including ghost layers
    //{
        //for (int j = 0; j < field->ny; j++) // Loop over all columns, including ghost layers
        //{
            //std::cout << field->data[(i * stride) + j] << " ";
        //}
        //std::cout << std::endl;
    //}
//}

int main(int argc, char **argv)
{
    field field;

    const uindex_t rows = atoi(argv[1]);
    const uindex_t cols = atoi(argv[2]);
    uindex_t nsteps = atoi(argv[3]);

    auto start = wall_clock_t::now();

    initialize(&field, rows, cols);

    auto stop = wall_clock_t::now();

    std::chrono::duration<float> elapsed = stop - start;
    printf("Field generation took %.3f seconds.\n", elapsed.count());

    // Diffusion constant
    double a = 0.5;

    // Compute the largest stable time step
    double dx2 = field.dx * field.dx;
    double dy2 = field.dy * field.dy;
    // Time step
    double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    Grid<double> grid = read(&field, rows, cols);

#if defined(STENCILSTREAM_TARGET_FPGA)
    sycl::device device(sycl::ext::intel::fpga_selector_v);
#else
    sycl::device device;
#endif

    StencilUpdate<HeatKernel> update({
        .transition_function = HeatKernel{a, dt, rows, cols},
        .n_iterations = nsteps,
        .device = device,
        .blocking = true,
    });
    grid = update(grid);

    std::cout << "Ending simulation" << std::endl;
    std::cout << "Walltime: " << update.get_walltime() << " s" << std::endl;


    write(grid); 

    return 0;
}