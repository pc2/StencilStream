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
#include <CL/sycl.hpp>
#include <StencilStream/Concepts.hpp>
#include <StencilStream/tdv/DevicePrecomputeSupplier.hpp>
#include <StencilStream/tdv/HostPrecomputeSupplier.hpp>
#include <StencilStream/tdv/InlineSupplier.hpp>
#include <StencilStream/tdv/NoneSupplier.hpp>
#include <catch2/catch_all.hpp>
#include <ext/intel/fpga_extensions.hpp>

using namespace stencil;

template <tdv::HostState TDVS, typename F> class TDVSTester {
  public:
    using KernelArgument = typename TDVS::KernelArgument;
    using LocalState = typename KernelArgument::LocalState;
    using Value = typename LocalState::Value;

    TDVSTester(TDVS tdvs, F verification_function)
        : tdvs(tdvs), verification_function(verification_function),
          test_queue(cl::sycl::ext::intel::fpga_emulator_selector_v) {}

    void run_test() {
        // Single pass
        tdvs.prepare_range(0, 32);
        run_pass(0, 32);

        // Multiple passes
        tdvs.prepare_range(32, 64);
        run_pass(32, 32);
        run_pass(64, 32);

        // Back in time
        tdvs.prepare_range(0, 64);
        run_pass(0, 32);
        run_pass(32, 32);

        // Overlapping, non-aligning passes
        tdvs.prepare_range(1, 64);
        run_pass(1, 32);
        run_pass(17, 32);
        run_pass(33, 32);

        // Incomplete passes
        tdvs.prepare_range(0, 32);
        run_pass(0, 16);
        run_pass(16, 16);

        // Single value pass
        tdvs.prepare_range(0, 1);
        run_pass(0, 1);
    }

  private:
    void run_pass(uindex_t i_generation, uindex_t n_generations) {
        cl::sycl::buffer<bool, 1> result_buffer = cl::sycl::range<1>(n_generations);

        test_queue.submit([&](cl::sycl::handler &cgh) {
            KernelArgument arg = tdvs.build_kernel_argument(cgh, i_generation, n_generations);
            auto result_ac =
                result_buffer.template get_access<cl::sycl::access::mode::discard_write>(cgh);
            auto verification_function = this->verification_function;

            cgh.single_task([=]() {
                LocalState local_state = arg.build_local_state();
                for (uindex_t i = 0; i < n_generations; i++) {
                    result_ac[i] =
                        verification_function(local_state.get_value(i), i_generation + i);
                }
            });
        });

        {
            auto result_ac = result_buffer.get_access<cl::sycl::access::mode::read>();
            for (uindex_t i = 0; i < n_generations; i++) {
                REQUIRE(result_ac[i]);
            }
        }
    }

    TDVS tdvs;
    F verification_function;
    cl::sycl::queue test_queue;
};

struct IndexValueFunction {
    using Value = uindex_t;

    uindex_t operator()(uindex_t i_generation) const { return i_generation + 1; }
};

static constexpr auto verify_index = [](uindex_t tdv, uindex_t i_generation) {
    return tdv == i_generation + 1;
};

static constexpr uindex_t max_n_generations = 32;

TEST_CASE("tdv::DevicePrecomputeSupplier", "[TimeDependentValueSupplier]") {
    tdv::DevicePrecomputeSupplier<IndexValueFunction, max_n_generations> supplier(
        IndexValueFunction{});
    TDVSTester tester(supplier, verify_index);
    tester.run_test();
}

TEST_CASE("tdv::InlineSupplier", "[TimeDependentValueSupplier]") {
    tdv::InlineSupplier<IndexValueFunction> supplier(IndexValueFunction{});
    TDVSTester tester(supplier, verify_index);
    tester.run_test();
}

TEST_CASE("tdv::HostPrecomputeSupplier", "[TimeDependentValueSupplier]") {
    tdv::HostPrecomputeSupplier<IndexValueFunction, max_n_generations> supplier(
        IndexValueFunction{});
    TDVSTester tester(supplier, verify_index);
    tester.run_test();
}

TEST_CASE("tdv::NoneSupplier", "[TimeDependentValueSupplier]") {
    // Nothing really to check with the none supplier.
    // This is more a test to check whether it compiles or if it causes other issues.
    TDVSTester tester(tdv::NoneSupplier(), [](std::monostate, uindex_t) { return true; });
    tester.run_test();
}