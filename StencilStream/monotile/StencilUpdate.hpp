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
#pragma once
#include "Grid.hpp"
#include "internal/StencilUpdateDesign.hpp"
#include <chrono>
#include <pc2/queue_extensions.hpp>
#include <type_traits>

namespace stencil {
namespace monotile {

/**
 * \brief Selects the inter-device communication mode for a StencilUpdate instance.
 *
 * When more than one FPGA is involved in a computation (e.g., via MPI-based multi-FPGA execution),
 * this enum controls how grids are transferred between devices between kernel invocations.
 */
enum class Connectivity {
    /// \brief A single FPGA handles the entire computation; no inter-device communication.
    SINGLE_DEVICE,
    /// \brief Multiple FPGAs are connected via Intel I/O pipes; data is streamed between devices.
    IO_PIPES,
};

/**
 * \brief A grid updater that applies an iterative stencil code to a grid.
 *
 * This updater applies an iterative stencil code, defined by the template parameter `F`, to the
 * grid; As often as requested. Since the underlying FPGA design follows the Monotile architecture
 * (See \ref monotile), an instance of this updater template can only process grids up to the
 * defined `max_grid_width` and `max_grid_height`.
 *
 * \tparam F The transition function to apply to input grids.
 *
 * \tparam temporal_parallelism (Optimization parameter) The number of iterations to compute in
 * parallel. Increasing this parameter leads to a higher performance, but it will also increase the
 * resource and space usage of the design. Excessive values may also decrease the clock frequency.
 * Also notice that subiterations within one iteration are always computed in parallel.
 *
 * \tparam spatial_parallelism (Optimization parameter) The number of cells to update in parallel
 * within one iteration. Increasing this parameter leads to a higher performance as long as the
 * product of the parameter and the size of the cell is below the physical memory word width. Common
 * values are 512 bits or 64 bytes.
 *
 * \tparam max_grid_height (Optimization parameter) The maximally supported grid height. For best
 * hardware utilization, this should be a power of two. Increase this parameter to the maximum your
 * application is expected to handle. However, higher maximal grid height might lead to increased
 * logic and space usage as well as decreased clock frequencies.
 *
 * \tparam max_grid_width (Optimization parameter) The maximally supported grid width. Increase
 * this parameter to the maximum your application is expected to handle. However, increasing the
 * maximal grid width will increase the BRAM usage of each PE.
 *
 * \tparam TDVStrategy (Optimization parameter) The precomputation strategy for the time-dependent
 * value system.
 *
 * \tparam word_size (Optimization parameter) The width of the global memory channel, in bytes. For
 * DDR-based systems, this should be 512 bits, or 64 bytes.
 */
template <concepts::TransitionFunction F, std::size_t temporal_parallelism = 1,
          std::size_t spatial_parallelism = 1, std::size_t max_grid_height = 1024,
          std::size_t max_grid_width = 1024, std::size_t n_kernels = 1,
          tdv::single_pass::Strategy<F, temporal_parallelism> TDVStrategy =
              tdv::single_pass::InlineStrategy,
          Connectivity connectivity = Connectivity::SINGLE_DEVICE>
class StencilUpdate {
  private:
    using Cell = F::Cell;
    using LocalDesign =
        internal::LocalStencilUpdateDesign<F, temporal_parallelism, spatial_parallelism,
                                           max_grid_height, max_grid_width, n_kernels, TDVStrategy>;
    using IOPipeDesign =
        internal::IOPipeStencilUpdateDesign<F, temporal_parallelism, spatial_parallelism,
                                            max_grid_height, max_grid_width, n_kernels,
                                            TDVStrategy>;
    using Design =
        std::conditional<connectivity == Connectivity::IO_PIPES, IOPipeDesign, LocalDesign>::type;

  public:
    /// \brief Shorthand for the used and supported grid type.
    using GridImpl = Grid<typename F::Cell, spatial_parallelism>;

    /**
     * \brief Parameters for the stencil updater.
     */
    struct Params {
        /**
         * \brief An instance of the transition function type.
         *
         * User applications may store runtime parameters here.
         */
        F transition_function;

        /**
         *  \brief The cell value to present for cells outside of the grid.
         */
        Cell halo_value = Cell();

        /**
         * \brief The iteration index offset.
         *
         * This offset will be added to the "actual" iteration index. This way, simulations can
         * "resume" with the next timestep if the intermediate grid has been evaluated by the host.
         */
        std::size_t iteration_offset = 0;

        /**
         * \brief The number of iterations to compute.
         */
        std::size_t n_iterations = 1;

        /**
         * \brief Should the stencil updater block until completion, or return immediately after all
         * kernels have been submitted.
         *
         * Choosing one option or the other won't effect the correctness: For example, if you choose
         * a non-blocking stencil updater and immediately try to access the grid after the updater
         * has returned, SYCL/OneAPI will block your thread until the computations are complete and
         * it can actually provide you access to the data.
         */
        bool blocking = false;

        /**
         * \brief Enable profiling.
         *
         * Setting this option to true will enable the recording of computation start and end
         * timestamps. The recorded kernel runtime can be fetched using the \ref
         * StencilUpdate::get_kernel_runtime method.
         */
        bool profiling = false;
    };

    /**
     * \brief Create a new stencil updater object.
     */
    StencilUpdate(Params params) : params(params), queues(), n_processed_cells(0), walltime(0.0) {
#if defined(STENCILSTREAM_TARGET_FPGA_EMU)
        std::function<int(const sycl::device &)> device_selector =
            sycl::ext::intel::fpga_emulator_selector_v;
#elif defined(STENCILSTREAM_TARGET_FPGA)
        std::function<int(const sycl::device &)> device_selector =
            sycl::ext::intel::fpga_selector_v;
#else
    #error                                                                                         \
        "Please link with the `StencilStream_Monotile`, `StencilStream_MonotileEmulator`, or `StencilStream_MonotileReport` targets!"
#endif

        if (connectivity == Connectivity::IO_PIPES) {
            queues = sycl::ext::pc2::mpi_queues<decltype(device_selector), sycl::property_list>(
                device_selector, Design::n_kernels, {sycl::property::queue::in_order{}});
        } else {
            sycl::device device(device_selector);
            for (std::size_t i_queue = 0; i_queue < Design::n_kernels; i_queue++) {
                queues.push_back(sycl::queue(device, {sycl::property::queue::in_order{}}));
            }
        }
    }

    /**
     * \brief Return a reference to the parameters.
     *
     * Modifications to the parameters struct will be used in the next call to \ref operator()().
     */
    Params &get_params() { return params; }

    /**
     * \brief Compute a new grid based on the source grid, using the configured transition function.
     *
     * The computation does not work in-place. Instead, it will allocate two additional grids with
     * the same size as the source grid and use them for a double buffering scheme. Therefore, you
     * are free to reuse the source grid as it will not be altered.
     *
     * If \ref Params::blocking is set to true, this method will block until the computation is
     * complete. Otherwise, it will return as soon as all kernels are submitted.
     */
    GridImpl operator()(GridImpl &source_grid) {
        using namespace stencil::internal;

        int mpi_initialized;
        MPI_Initialized(&mpi_initialized);

        if (source_grid.get_grid_width() < 2) {
            throw std::range_error("The grid is too narrow. The monotile backend can only process "
                                   "grids with at least two columns.");
        }
        if (source_grid.get_grid_height() > max_grid_height) {
            throw std::range_error("The grid is too tall for the stencil update kernel.");
        }
        if (source_grid.get_grid_width() > max_grid_width) {
            throw std::range_error("The grid is too wide for the stencil update kernel.");
        }

        Design design(params.transition_function, params.halo_value, params.iteration_offset,
                      params.n_iterations, queues);

        if (mpi_initialized) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
        auto walltime_start = std::chrono::high_resolution_clock::now();

        GridImpl result_grid =
            design.submit_simulation(source_grid, params.iteration_offset, params.n_iterations);
        if (params.blocking) {
            design.wait_for_queues();
        }

        if (mpi_initialized) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
        auto walltime_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> walltime = walltime_end - walltime_start;
        this->walltime += walltime.count();

        n_processed_cells +=
            params.n_iterations * source_grid.get_grid_height() * source_grid.get_grid_width();

        return result_grid;
    }

    /**
     * \brief Return the accumulated total number of cells processed by this updater.
     *
     * For each call of to \ref operator()(), this is the width times the height of the grid, times
     * the number of computed iterations. This will also be accumulated across multiple calls to
     * \ref operator()().
     */
    std::size_t get_n_processed_cells() const { return n_processed_cells; }

    /**
     * \brief Return the accumulated total runtime of the execution kernel.
     *
     * This runtime is accumulated across multiple calls to \ref operator()(). However, this is only
     * possible if \ref Params::profiling is set to true.
     */
    [[deprecated("Not implemented, equal to walltime")]] double get_kernel_runtime() const {
        return walltime;
    }

    /**
     * \brief Return the accumulated runtime of the updater, measured from the host side.
     *
     * For each call to \ref operator()(), the time it took to submit all kernels and, if \ref
     * Params::blocking is true, to finish the computation is recorded and accumulated.
     */
    double get_walltime() const { return walltime; }

  private:
    Params params;
    std::vector<sycl::queue> queues;
    std::size_t n_processed_cells;
    double walltime;
};

} // namespace monotile
} // namespace stencil