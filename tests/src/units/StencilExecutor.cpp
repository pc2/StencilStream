
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
#include <StencilStream/MonotileExecutor.hpp>
#include <StencilStream/StencilExecutor.hpp>
#include <StencilStream/SimpleCPUExecutor.hpp>
#include <res/TransFuncs.hpp>
#include <res/catch.hpp>
#include <res/constants.hpp>

using namespace std;
using namespace stencil;
using namespace cl::sycl;

using TransFunc = FPGATransFunc<stencil_radius>;
using SingleContextExecutorImpl = SingleContextExecutor<Cell, stencil_radius, TransFunc>;
using StencilExecutorImpl = StencilExecutor<Cell, stencil_radius, TransFunc, pipeline_length>;
using MonotileExecutorImpl = MonotileExecutor<Cell, stencil_radius, TransFunc, pipeline_length>;
using SimpleCPUExecutorImpl = SimpleCPUExecutor<Cell, stencil_radius, TransFunc>;

void test_executor_set_input_copy_output(SingleContextExecutorImpl *executor, uindex_t grid_width,
                                         uindex_t grid_height) {
    buffer<Cell, 2> in_buffer(range<2>(grid_width, grid_height));
    {
        auto in_buffer_ac = in_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                in_buffer_ac[c][r] = Cell{index_t(c), index_t(r), 0, CellStatus::Normal};
            }
        }
    }

    executor->set_input(in_buffer);

    buffer<Cell, 2> out_buffer(range<2>(grid_width, grid_height));
    executor->copy_output(out_buffer);

    auto out_buffer_ac = out_buffer.get_access<access::mode::read>();
    for (uindex_t c = 0; c < grid_width; c++) {
        for (uindex_t r = 0; r < grid_height; r++) {
            REQUIRE(out_buffer_ac[c][r].c == c);
            REQUIRE(out_buffer_ac[c][r].r == r);
            REQUIRE(out_buffer_ac[c][r].i_generation == 0);
            REQUIRE(out_buffer_ac[c][r].status == CellStatus::Normal);
        }
    }
}

TEST_CASE("StencilExecutor::copy_output(cl::sycl::buffer<T, 2>)", "[StencilExecutor]") {
    StencilExecutorImpl executor(Cell::halo(), TransFunc());
    test_executor_set_input_copy_output(&executor, grid_width, grid_height);
}

TEST_CASE("MonotileExecutor::copy_output(cl::sycl::buffer<T, 2>)", "[MonotileExecutor]") {
    MonotileExecutorImpl executor(Cell::halo(), TransFunc());
    test_executor_set_input_copy_output(&executor, tile_width - 1, tile_height - 1);
}

TEST_CASE("SimpleCPUExecutor::copy_output(cl::sycl::buffer<T, 2>)", "[SimpleCPUExecutor]") {
    SimpleCPUExecutorImpl executor(Cell::halo(), TransFunc());
    test_executor_set_input_copy_output(&executor, tile_width - 1, tile_height - 1);
}

void test_executor_run(SingleContextExecutorImpl *executor, uindex_t grid_width,
                       uindex_t grid_height) {
    executor->select_emulator();
    
    uindex_t n_generations = 2 * pipeline_length + 1;

    buffer<Cell, 2> in_buffer(range<2>(grid_width, grid_height));
    {
        auto in_buffer_ac = in_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                in_buffer_ac[c][r] = Cell{index_t(c), index_t(r), 0, CellStatus::Normal};
            }
        }
    }

    REQUIRE(executor->get_i_generation() == 0);

    executor->set_input(in_buffer);
    REQUIRE(executor->get_grid_range().c == grid_width);
    REQUIRE(executor->get_grid_range().r == grid_height);

    executor->run(n_generations);
    REQUIRE(executor->get_i_generation() == n_generations);

    buffer<Cell, 2> out_buffer(range<2>(grid_width, grid_height));
    executor->copy_output(out_buffer);

    {
        auto out_buffer_ac = out_buffer.get_access<access::mode::read>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                REQUIRE(out_buffer_ac[c][r].c == c);
                REQUIRE(out_buffer_ac[c][r].r == r);
                REQUIRE(out_buffer_ac[c][r].i_generation == n_generations);
                REQUIRE(out_buffer_ac[c][r].status == CellStatus::Normal);
            }
        }
    }

    // Now, a second run to show that behavior is still correct when i_generation != 0:
    executor->run(n_generations);
    REQUIRE(executor->get_i_generation() == 2 * n_generations);

    executor->copy_output(out_buffer);
    {
        auto out_buffer_ac = out_buffer.get_access<access::mode::read>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                REQUIRE(out_buffer_ac[c][r].c == c);
                REQUIRE(out_buffer_ac[c][r].r == r);
                REQUIRE(out_buffer_ac[c][r].i_generation == 2 * n_generations);
                REQUIRE(out_buffer_ac[c][r].status == CellStatus::Normal);
            }
        }
    }
}

TEST_CASE("StencilExecutor::run", "[StencilExecutor]") {
    StencilExecutorImpl executor(Cell::halo(), TransFunc());
    test_executor_run(&executor, grid_width, grid_height);
}

TEST_CASE("MonotileExecutor::run", "[MonotileExecutor]") {
    MonotileExecutorImpl executor(Cell::halo(), TransFunc());
    test_executor_run(&executor, grid_width, grid_height);
}

TEST_CASE("SimpleCPUExecutor::run", "[SimpleCPUExecutor]") {
    SimpleCPUExecutorImpl executor(Cell::halo(), TransFunc());
    test_executor_run(&executor, grid_width, grid_height);
}