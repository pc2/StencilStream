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
#include "Index.hpp"

namespace stencil {
/**
 * \brief Collection of runtime performance information for a \ref SingleQueueExecutor.
 */
class RuntimeSample {
  public:
    /**
     * \brief Prepare the collection of SYCL events.
     */
    RuntimeSample() : makespan(0.0), n_passes(0.0) {}

    /**
     * \brief Retrieve the starting timestamp of the event, in seconds.
     *
     * This will fail with an exception if the event's queue has not been configured to collect
     * runtime information.
     */
    static double start_of_event(cl::sycl::event event) {
        return double(event.get_profiling_info<cl::sycl::info::event_profiling::command_start>()) /
               timesteps_per_second;
    }

    /**
     * \brief Retrieve the ending timestamp of the event, in seconds.
     *
     * This will fail with an exception if the event's queue has not been configured to collect
     * runtime information.
     */
    static double end_of_event(cl::sycl::event event) {
        return double(event.get_profiling_info<cl::sycl::info::event_profiling::command_end>()) /
               timesteps_per_second;
    }

    /**
     * \brief Retrieve the runtime/makespan of the event, in seconds.
     *
     * This will fail with an exception if the event's queue has not been configured to collect
     * runtime information.
     */
    static double runtime_of_event(cl::sycl::event event) {
        return end_of_event(event) - start_of_event(event);
    }

    /**
     * \brief Add the runtime/makespan of a single pass over the grid to the database.
     */
    void add_pass(double pass_runtime) {
        makespan += pass_runtime;
        n_passes += 1.0;
    }

    /**
     * \brief Add the event of a single pass over the grid to the database.
     */
    void add_pass(cl::sycl::event event) { add_pass(runtime_of_event(event)); }

    /**
     * \brief Get the makespan of all grid passes.
     *
     * \return The makespan in seconds it took to execute all passes.
     */
    double get_total_runtime() { return makespan; }

    /**
     * \brief Calculate the mean execution speed in passes per second.
     *
     * \return The mean execution speed in passes per second.
     */
    double get_mean_speed() { return makespan / n_passes; }

  private:
    static constexpr double timesteps_per_second = 1000000000.0;

    double makespan, n_passes;
};
} // namespace stencil