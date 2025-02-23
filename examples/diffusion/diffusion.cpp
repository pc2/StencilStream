#if defined(STENCIL_STREAM_BACKEND_CPU)
#include <StencilStream/cpu/StencilUpdate.hpp>
#elif defined(STENCILSTREAM_BACKEND_CUDA)
#include <StencilStream/cuda/StencilUpdate.hpp>
#endif

#include <StencilStream/BaseTransitionFunction.hpp>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace stencil;

#if defined(STENCILSTREAM_BACKEND_CPU)
using namespace stencil::cpu;
#elif (STENCILSTREAM_BACKEND_CUDA)
using namespace stencil::cuda;
#endif

// Definition of the grid
struct field {
    int nx, ny;
    double dx, dy;
    std::vector<double> data;
};

constexpr auto DX = 0.01;
constexpr auto DY = 0.01;

struct HeatKernel : public BaseTransitionFunction {

    using Cell = double;

    double a, dt;
    size_t rows, cols;

    HeatKernel(double a_, double dt_, size_t rows_, size_t cols_)
        : a(a_), dt(dt_), rows(rows_), cols(cols_) {}

    double operator()(Stencil<double, 1> const &stencil) const {
        auto r = stencil.id[0];
        auto c = stencil.id[1];

        // halo_value conditions
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
            double center = stencil[0][0];
            double left = stencil[-1][0];
            double right = stencil[1][0];
            double top = stencil[0][-1];
            double bottom = stencil[0][1];

            // Calculation
            return center + a * dt *
                                ((left - 2.0 * center + right) / (DX * DX) +
                                 (top - 2.0 * center + bottom) / (DY * DY));
        }
    }
};

// Generation of the field with the boundary conditions
void initialize(field *field, size_t rows, size_t cols) {
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
    for (int i = 0; i < field->nx; i++) {
        for (int j = 0; j < field->ny; j++) {
            ind = i * field->ny + j;

            dx = i - field->nx / 2;
            dy = j - field->ny / 2;
            if (dx * dx + dy * dy < radius * radius) {
                field->data[ind] = 5.0;
            } else {
                field->data[ind] = 65.0;
            }
        }
    }

    // Boundary conditions
    for (int i = 0; i < field->nx; i++) {
        field->data[i * field->ny + field->ny - 1] = 70.0;
        field->data[i * field->ny] = 20.0;
    }

    for (int j = 0; j < field->ny; j++) {
        field->data[j] = 85.0;
    }
    for (int j = 0; j < field->ny; j++) {
        field->data[(field->nx - 1) * field->ny + j] = 5.0;
    }
}

void write(Grid<double> output_grid, size_t iteration, std::string folder) {
    std::string name = folder + "/" + to_string(iteration) + ".csv";
    std::ofstream myFile(name);

    Grid<double>::GridAccessor<sycl::access::mode::read> grid_ac(output_grid);

    for (size_t r = 0; r < output_grid.get_grid_height(); r++) {
        for (size_t c = 0; c < output_grid.get_grid_width(); c++) {
            myFile << grid_ac[r][c];
            if (c + 1 != output_grid.get_grid_width()) {
                myFile << ",";
            }
        }
        if (r + 1 != output_grid.get_grid_height()) {
            myFile << std::endl;
        }
    }
}

Grid<double> read(field *field, size_t rows, size_t cols) {
    Grid<double> input_grid(rows, cols);
    {
        Grid<double>::GridAccessor<sycl::access::mode::read_write> grid_ac(input_grid);

        size_t stride = input_grid.get_grid_height();
        for (size_t r = 0; r < cols; r++) {
            for (size_t c = 0; c < rows; c++) {
                grid_ac[r][c] = field->data[c * stride + r];
            }
        }
    }
    return input_grid;
}

int main(int argc, char **argv) {
    // Walltime counter start
    auto walltime_start = std::chrono::high_resolution_clock::now();

    field field;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <width> <height> <n_iterations>" << std::endl;
        return 1;
    }

    const size_t rows = stoi(argv[1]);
    const size_t cols = stoi(argv[2]);
    size_t nsteps = stoi(argv[3]);

    // Field generation timer start
    auto field_geration_time_start = std::chrono::high_resolution_clock::now();

    initialize(&field, rows, cols);

    auto field_generation_time_end = std::chrono::high_resolution_clock::now();

    auto field_generation_time =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            field_generation_time_end - field_geration_time_start);
    std::cout << "Field generation took = " << field_generation_time.count() << " ms" << std::endl;

    // Diffusion constant
    double a = 0.5;

    // Compute the largest stable time step
    double dx2 = field.dx * field.dx;
    double dy2 = field.dy * field.dy;

    // Time step
    double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));

    Grid<double> grid = read(&field, rows, cols);

#if defined(STENCILSTREAM_TARGET_CUDA)
    sycl::device device(sycl::gpu_selector_v);
#else
    sycl::device device;
#endif

    // Output folder definition
    std::string outputFolder = "output";
    std::filesystem::create_directories(outputFolder);

    // Kerneltime variable for counting the single kernel times
    auto temp = 0.0;

    for (size_t it = 0; nsteps >= it; it++) {
        // Kernel time start
        auto kernel_start_time = std::chrono::high_resolution_clock::now();

        StencilUpdate<HeatKernel> heat_kernel_update({
            .transition_function = HeatKernel{a, dt, rows, cols},
            .n_iterations = 1,
            .device = device,
            .blocking = true,
        });
        auto kernel_end_time = std::chrono::high_resolution_clock::now();

        auto kernel_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            kernel_end_time - kernel_start_time);

        // Kenrel time summation
        temp += kernel_time.count();

        grid = heat_kernel_update(grid);

        // Write grid to csv file
        write(grid, it, outputFolder);

        if (it % 10 == 0) {
            std::cout << "Iteration: " << it << std::endl;
        }
    }

    auto walltime_end = std::chrono::high_resolution_clock::now();

    auto walltime =
        std::chrono::duration_cast<std::chrono::duration<double>>(walltime_end - walltime_start);

    std::cout << "Kerneltime: " << temp << " ms" << std::endl;
    std::cout << "Walltime: " << walltime.count() << " s" << std::endl;

    return 0;
}