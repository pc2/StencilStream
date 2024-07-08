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
#include "Kernel.hpp"
#include <deque>
#include <sycl/ext/intel/fpga_extensions.hpp>

#if MATERIAL == 0
    #include "material/CoefResolver.hpp"
using MaterialResolver = CoefResolver;
#elif MATERIAL == 1
    #include "material/LUTResolver.hpp"
using MaterialResolver = LUTResolver;
#elif MATERIAL == 2
    #include "material/RenderResolver.hpp"
using MaterialResolver = RenderResolver;
#endif

using KernelImpl = Kernel<MaterialResolver>;
using CellImpl = KernelImpl::Cell;

#include <StencilStream/tdv/SinglePassStrategies.hpp>

#if TDVS_TYPE == 0
using TDVStrategy = tdv::single_pass::InlineStrategy;
#elif TDVS_TYPE == 1
using TDVStrategy = tdv::single_pass::PrecomputeOnDeviceStrategy;
#elif TDVS_TYPE == 2
using TDVStrategy = tdv::single_pass::PrecomputeOnHostStrategy;
#endif

#if defined(STENCILSTREAM_BACKEND_MONOTILE)
    #include <StencilStream/monotile/StencilUpdate.hpp>

using Grid = monotile::Grid<CellImpl>;
using StencilUpdate = monotile::StencilUpdate<KernelImpl, n_processing_elements, tile_width,
                                              tile_height, TDVStrategy>;
#elif defined(STENCILSTREAM_BACKEND_TILING)
    #include <StencilStream/tiling/StencilUpdate.hpp>
using StencilUpdate =
    tiling::StencilUpdate<KernelImpl, n_processing_elements, tile_width, tile_height, TDVStrategy>;
using Grid = StencilUpdate::GridImpl;
#elif defined(STENCILSTREAM_BACKEND_CPU)
    #include <StencilStream/cpu/StencilUpdate.hpp>
using Grid = cpu::Grid<CellImpl>;
using StencilUpdate = cpu::StencilUpdate<KernelImpl>;
#endif

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

enum class CellField {
    EX,
    EY,
    HZ,
    HZ_SUM,
};

void save_frame(Grid frame_buffer, uindex_t iteration_index, CellField field,
                Parameters const &parameters) {
    Grid::GridAccessor<access::mode::read> frame(frame_buffer);

    ostringstream frame_path;
    frame_path << parameters.out_dir << "/";
    switch (field) {
    case CellField::EX:
        frame_path << "ex";
        break;
    case CellField::EY:
        frame_path << "ey";
        break;
    case CellField::HZ:
        frame_path << "hz";
        break;
    case CellField::HZ_SUM:
        frame_path << "hz_sum";
        break;
    default:
        break;
    }
    frame_path << "." << iteration_index << ".csv";
    std::ofstream out(frame_path.str());

    for (uindex_t r = 0; r < parameters.grid_range()[1]; r++) {
        for (uindex_t c = 0; c < parameters.grid_range()[0]; c++) {
            switch (field) {
            case CellField::EX:
                out << frame[c][r].cell.ex;
                break;
            case CellField::EY:
                out << frame[c][r].cell.ey;
                break;
            case CellField::HZ:
                out << frame[c][r].cell.hz;
                break;
            case CellField::HZ_SUM:
                out << frame[c][r].cell.hz_sum;
                break;
            default:
                break;
            }

            if (c != parameters.grid_range()[0] - 1) {
                out << ",";
            }
        }
        if (r != parameters.grid_range()[1] - 1) {
            out << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    Parameters parameters(argc, argv);
    parameters.print_configuration();

#if defined(STENCILSTREAM_BACKEND_MONOTILE)
    if (parameters.grid_range()[0] > tile_width || parameters.grid_range()[1] > tile_height) {
        std::cerr << "Error: The grid may not exceed the size of the tile (" << tile_width << " by "
                  << tile_height << " cells) when using the monotile architecture." << std::endl;
        exit(1);
    }
#endif

    MaterialResolver mat_resolver(parameters);

    Grid grid(parameters.grid_range());
    {
        Grid::GridAccessor<access::mode::read_write> init_ac(grid);
        for (uindex_t c = 0; c < parameters.grid_range()[0]; c++) {
            for (uindex_t r = 0; r < parameters.grid_range()[1]; r++) {
                float a = float(c) - float(parameters.grid_range()[0]) / 2.0;
                float b = float(r) - float(parameters.grid_range()[1]) / 2.0;
                float distance = parameters.dx * sqrt(a * a + b * b);

                float radius = 0.0;
                for (uindex_t i = 0; i <= parameters.rings.size(); i++) {
                    if (i < parameters.rings.size()) {
                        radius += parameters.rings[i].width;
                        if (distance < radius) {
                            init_ac[c][r] = CellImpl::from_parameters(parameters, i);
                            break;
                        }
                    } else {
                        init_ac[c][r] = CellImpl::from_parameters(parameters, i);
                    }
                }
            }
        }
    }

#if defined(STENCILSTREAM_TARGET_FPGA)
    sycl::device device(sycl::ext::intel::fpga_selector_v);
#else
    sycl::device device;
#endif

    StencilUpdate simulation({
        .transition_function = KernelImpl(parameters, mat_resolver), .halo_value = CellImpl::halo(),
        .iteration_offset = 0, .n_iterations = parameters.n_timesteps(), .device = device,
        .blocking = true, // enable blocking for meaningful walltime measurements
#if !defined(STENCILSTREAM_BACKEND_CPU)
            .profiling = true, // enable additional profiling for FPGA targets
#endif
    });

    uindex_t n_timesteps = parameters.n_timesteps();
    uindex_t last_saved_iteration = 0;

    std::cout << "Simulating..." << std::endl;

    if (parameters.n_snap_timesteps().has_value()) {
        uindex_t n_snap_timesteps = parameters.n_snap_timesteps().value();
        simulation.get_params().n_iterations = n_snap_timesteps;
        for (uindex_t &i = simulation.get_params().iteration_offset; i < parameters.n_timesteps();
             i += n_snap_timesteps) {
            grid = simulation(grid);
            save_frame(grid, i + n_snap_timesteps, CellField::HZ, parameters);
        }
    } else {
        grid = simulation(grid);
    }

    std::cout << "Simulation complete!" << std::endl;
    std::cout << "Walltime: " << simulation.get_walltime() << " s" << std::endl;
#if !defined(STENCILSTREAM_BACKEND_CPU)
    // Print pure kernel runtime for FPGA targets
    std::cout << "Kernel Runtime: " << simulation.get_kernel_runtime() << " s" << std::endl;
#endif

    save_frame(grid, n_timesteps, CellField::HZ_SUM, parameters);

    return 0;
}
