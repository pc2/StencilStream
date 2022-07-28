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
#pragma once
#include "AbstractExecutor.hpp"
#include <ext/intel/fpga_extensions.hpp>
#include <optional>

namespace stencil {
/**
 * \brief An abstract executor with common code for executors that work with a single SYCL queue.
 *
 * This class contains common code for executors that work with execution kernels running on a
 * single queue. This includes queue selection and management as well as event-based runtime
 * analysis.
 *
 * User code that may work with any kind of executor should use pointers to the general \ref
 * AbstractExecutor.
 *
 * \tparam TransFunc The type of the transition function.
 */
template <TransitionFunction TransFunc>
class SingleContextExecutor : public AbstractExecutor<TransFunc> {
  public:
    using Cell = typename TransFunc::Cell;

    /**
     * \brief Create a new executor.
     * \param halo_value The value of cells that are outside the grid.
     * \param trans_func The instance of the transition function that should be used to calculate
     * new generations.
     */
    SingleContextExecutor(Cell halo_value, TransFunc trans_func)
        : AbstractExecutor<TransFunc>(halo_value, trans_func), device(std::nullopt),
          context(std::nullopt) {}

    /**
     * \brief Return the configured queue.
     *
     * If no queue has been configured yet, this method will configure and return a queue targeting
     * the FPGA emulator, without runtime analysis.
     */
    cl::sycl::queue new_queue(bool in_place = false) {
        cl::sycl::property_list queue_properties;
        if (in_place) {
            queue_properties = {cl::sycl::property::queue::enable_profiling{},
                                cl::sycl::property::queue::in_order{}};
        } else {
            queue_properties = {cl::sycl::property::queue::enable_profiling{}};
        }
        if (!this->device.has_value() || !this->context.has_value()) {
            this->select_emulator();
        }
        return cl::sycl::queue(*this->context, *this->device, queue_properties);
    }

    /**
     * \brief Manually set the SYCL queue to use for execution.
     *
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for
     * the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will
     * lead to exceptions.
     *
     * In order to use runtime analysis features, the queue has to be configured with the
     * `cl::sycl::property::queue::enable_profiling` property. A `std::runtime_error` is thrown if
     * `runtime_analysis` is true and the passed queue does not have this property.
     *
     * \deprecated This method is deprecated since the `runtime_analysis` flag is redundant by now.
     * Use the other variant without the `runtime_analysis` flag instead.
     *
     * \param queue The new SYCL queue to use for execution.
     * \param runtime_analysis Enable event-level runtime analysis.
     */
    [[deprecated("Use build_context() instead")]] void set_queue(cl::sycl::queue queue,
                                                                 bool runtime_analysis) {
        this->build_context(queue.get_device());
    }

    /**
     * \brief Manually set the SYCL queue to use for execution.
     *
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for
     * the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will
     * lead to exceptions.
     *
     * Runtime analysis is enabled by configuring the queue with the
     * `cl::sycl::property::queue::enable_profiling` property.
     *
     * \param queue The new SYCL queue to use for execution.
     */
    [[deprecated("Use build_context() instead")]] void set_queue(cl::sycl::queue queue) {
        this->build_context(queue.get_device());
    }

    /**
     * \brief Set up a SYCL queue with the FPGA emulator device and optional runtime analysis.
     *
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for
     * the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will
     * lead to exceptions.
     *
     * \param runtime_analysis Enable event-level runtime analysis.
     */
    [[deprecated("Use select_emulator() instead")]] void select_emulator(bool runtime_analysis) {
        this->select_emulator();
    }

    /**
     * \brief Set up a SYCL queue with an FPGA device and optional runtime analysis.
     *
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for
     * the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will
     * lead to exceptions.
     *
     * \param runtime_analysis Enable event-level runtime analysis.
     */
    [[deprecated("Use select_fpga() instead")]] void select_fpga(bool runtime_analysis) {
        this->select_fpga();
    }

    void select_cpu() { this->build_context(cl::sycl::cpu_selector().select_device()); }

    void select_emulator() {
        this->build_context(cl::sycl::ext::intel::fpga_emulator_selector().select_device());
    }

    void select_fpga() {
        this->build_context(cl::sycl::ext::intel::fpga_selector().select_device());
    }

    void build_context(cl::sycl::device device) {
        this->device = device;
        this->context = cl::sycl::context(device);
    }

  private:
    std::optional<cl::sycl::device> device;
    std::optional<cl::sycl::context> context;
};
} // namespace stencil