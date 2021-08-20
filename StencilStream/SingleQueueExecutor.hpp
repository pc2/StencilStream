/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
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
#include "RuntimeSample.hpp"
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <optional>

namespace stencil {
template <typename T, uindex_t stencil_radius, typename TransFunc, uindex_t pipeline_length = 1>
class SingleQueueExecutor : public AbstractExecutor<T, stencil_radius, TransFunc> {
  protected:
    virtual void run_pass(uindex_t target_i_generation) = 0;

  public:
    SingleQueueExecutor(T halo_value, TransFunc trans_func)
        : AbstractExecutor<T, stencil_radius, TransFunc>(halo_value, trans_func),
          queue(std::nullopt), runtime_sample(std::nullopt) {}

    void run(uindex_t n_generations) override {
        cl::sycl::queue queue = this->get_queue();

        uindex_t target_i_generation = n_generations + this->get_i_generation();
        uindex_t n_passes = n_generations / pipeline_length;
        if (n_generations % pipeline_length != 0) {
            n_passes += 1;
        }

        for (uindex_t i = 0; i < n_passes; i++) {
            this->run_pass(target_i_generation);
            this->inc_i_generation(
                std::min(target_i_generation - this->get_i_generation(), pipeline_length));
        }
    }

    cl::sycl::queue &get_queue() {
        if (!this->queue.has_value()) {
            select_emulator(false);
        }
        return *queue;
    }

    /**
     * \brief Manually set the SYCL queue to use for execution.
     *
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for
     * the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will
     * lead to exceptions.
     *
     * In order to use runtime analysis features, the queue has to be configured with the
     * `cl::sycl::property::queue::enable_profiling` property.
     *
     * \param queue The new SYCL queue to use for execution.
     * \param runtime_analysis Enable event-level runtime analysis.
     */
    void set_queue(cl::sycl::queue queue, bool runtime_analysis) {
        if (runtime_analysis &&
            !queue.has_property<cl::sycl::property::queue::enable_profiling>()) {
            throw std::runtime_error(
                "Runtime analysis is enabled, but the queue does not support it.");
        }
        this->queue = queue;
    }

    /**
     * \brief Set up a SYCL queue with the FPGA emulator device.
     *
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for
     * the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will
     * lead to exceptions.
     */
    void select_emulator() { select_emulator(false); }

    /**
     * \brief Set up a SYCL queue with an FPGA device.
     *
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for
     * the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will
     * lead to exceptions.
     */
    void select_fpga() { select_fpga(false); }

    /**
     * \brief Set up a SYCL queue with the FPGA emulator device and optional runtime analysis.
     *
     * Note that as of OneAPI Version 2021.1.1, device code is usually built either for CPU/GPU, for
     * the FPGA emulator or for a specific FPGA. Using the wrong queue with the wrong device will
     * lead to exceptions.
     *
     * \param runtime_analysis Enable event-level runtime analysis.
     */
    void select_emulator(bool runtime_analysis) {
        this->queue = cl::sycl::queue(cl::sycl::INTEL::fpga_emulator_selector(),
                                      get_queue_properties(runtime_analysis));
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
    void select_fpga(bool runtime_analysis) {
        this->queue = cl::sycl::queue(cl::sycl::INTEL::fpga_selector(),
                                      get_queue_properties(runtime_analysis));
    }

    bool is_runtime_analysis_enabled() const {
        if (queue.has_value()) {
            return queue->has_property<cl::sycl::property::queue::enable_profiling>();
        } else {
            return false;
        }
    }

    /**
     * \brief Return the runtime information collected from the last \ref StencilExecutor.run call.
     *
     * \return The collected runtime information. May be `nullopt` if no runtime analysis was
     * configured.
     */
    std::optional<RuntimeSample> get_runtime_sample() { return runtime_sample; }

  private:
    static cl::sycl::property_list get_queue_properties(bool runtime_analysis) {
        cl::sycl::property_list properties;
        if (runtime_analysis) {
            properties = {cl::sycl::property::queue::enable_profiling{}};
        } else {
            properties = {};
        }
        return properties;
    }

    std::optional<cl::sycl::queue> queue;
    std::optional<RuntimeSample> runtime_sample;
};
} // namespace stencil