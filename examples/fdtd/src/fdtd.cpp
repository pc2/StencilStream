/*
 * Copyright © 2020-2022 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "SourceFunction.hpp"
#include <deque>

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

#if SOURCE == 0
    #include <StencilStream/tdv/InlineSupplier.hpp>
using SourceSupplier = tdv::InlineSupplier<SourceFunction>;
#elif SOURCE == 1
    #include <StencilStream/tdv/HostPrecomputeSupplier.hpp>
using SourceSupplier = tdv::HostPrecomputeSupplier<SourceFunction, gens_per_pass>;
#endif

using KernelImpl = Kernel<MaterialResolver>;
using CellImpl = KernelImpl::Cell;

#if EXECUTOR == 0
    #include <StencilStream/MonotileExecutor.hpp>
using Executor =
    MonotileExecutor<KernelImpl, SourceSupplier, n_processing_elements, tile_width, tile_height>;
#elif EXECUTOR == 1
    #include <StencilStream/TilingExecutor.hpp>
using Executor =
    TilingExecutor<KernelImpl, SourceSupplier, n_processing_elements, tile_width, tile_height>;
#elif EXECUTOR == 2
    #include <StencilStream/SimpleCPUExecutor.hpp>
using Executor = SimpleCPUExecutor<KernelImpl, SourceSupplier>;
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

void save_frame(cl::sycl::buffer<CellImpl, 2> frame_buffer, uindex_t generation_index,
                CellField field, Parameters const &parameters) {
    auto frame = frame_buffer.get_access<access::mode::read>();

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
    frame_path << "." << generation_index << ".csv";
    std::ofstream out(frame_path.str());

    for (uindex_t r = 0; r < frame.get_range()[1]; r++) {
        for (uindex_t c = 0; c < frame.get_range()[0]; c++) {
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

            if (c != frame.get_range()[0] - 1) {
                out << ",";
            }
        }
        if (r != frame.get_range()[1] - 1) {
            out << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    Parameters parameters(argc, argv);
    parameters.print_configuration();

#if EXECUTOR == 0
    if (parameters.grid_range()[0] > tile_width || parameters.grid_range()[1] > tile_height) {
        std::cerr << "Error: The grid may not exceed the size of the tile (" << tile_width << " by "
                  << tile_height << " cells) when using the monotile architecture." << std::endl;
        exit(1);
    }
#endif

    MaterialResolver mat_resolver(parameters);

    cl::sycl::buffer<CellImpl, 2> grid_buffer(parameters.grid_range());
    {
        auto init_ac = grid_buffer.get_access<cl::sycl::access::mode::discard_write>();
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

    KernelImpl kernel(parameters, mat_resolver);

    SourceFunction source_function(parameters);
    SourceSupplier source_supplier(source_function);

    Executor executor(CellImpl::halo(), kernel, source_supplier);
    executor.set_input(grid_buffer);
#ifdef HARDWARE
    executor.select_fpga();
#endif

    uindex_t n_timesteps = parameters.n_timesteps();
    uindex_t last_saved_generation = 0;

    std::cout << "Simulating..." << std::endl;

    if (parameters.interval().has_value()) {
        uindex_t interval = parameters.interval().value();
        auto snapshot_handler = [&](cl::sycl::buffer<CellImpl, 2> cell_buffer,
                                    uindex_t i_generation) {
            save_frame(cell_buffer, i_generation, CellField::HZ, parameters);
        };
        executor.run_with_snapshots(n_timesteps, interval, snapshot_handler);
    } else {
        executor.run(n_timesteps);
    }

    std::cout << "Simulation complete!" << std::endl;
    std::cout << "Makespan: " << executor.get_runtime_sample().get_total_runtime() << " s"
              << std::endl;

    executor.copy_output(grid_buffer);
    save_frame(grid_buffer, n_timesteps, CellField::HZ_SUM, parameters);

    return 0;
}
