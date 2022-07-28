
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
#include <StencilStream/MonotileExecutor.hpp>
#include <StencilStream/SimpleCPUExecutor.hpp>
#include <StencilStream/TilingExecutor.hpp>
#include <res/TransFuncs.hpp>
#include <res/catch.hpp>
#include <res/constants.hpp>

using namespace std;
using namespace stencil;
using namespace cl::sycl;

using TransFunc = FPGATransFunc<stencil_radius>;
using SingleContextExecutorImpl = SingleContextExecutor<TransFunc>;
using TilingExecutorImpl =
    TilingExecutor<TransFunc, n_processing_elements, tile_width, tile_height>;
using MonotileExecutorImpl =
    MonotileExecutor<TransFunc, n_processing_elements, tile_width, tile_height>;
using SimpleCPUExecutorImpl = SimpleCPUExecutor<TransFunc>;

void test_executor_set_input_copy_output(SingleContextExecutorImpl *executor, uindex_t grid_width,
                                         uindex_t grid_height) {
    buffer<Cell, 2> in_buffer(range<2>(grid_width, grid_height));
    {
        auto in_buffer_ac = in_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                in_buffer_ac[c][r] = Cell{index_t(c), index_t(r), 0, 0, CellStatus::Normal};
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
            REQUIRE(out_buffer_ac[c][r].i_subgeneration == 0);
            REQUIRE(out_buffer_ac[c][r].status == CellStatus::Normal);
        }
    }
}

TEST_CASE("TilingExecutor::copy_output(cl::sycl::buffer<T, 2>)", "[TilingExecutor]") {
    TilingExecutorImpl executor(Cell::halo(), TransFunc());
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

void test_executor(SingleContextExecutorImpl *executor, uindex_t grid_width, uindex_t grid_height,
                   uindex_t n_generations) {
    buffer<Cell, 2> in_buffer(range<2>(grid_width, grid_height));
    {
        auto in_buffer_ac = in_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                in_buffer_ac[c][r] = Cell{index_t(c), index_t(r), 0, 0, CellStatus::Normal};
            }
        }
    }

    executor->set_input(in_buffer);
    executor->set_i_generation(0);
    executor->select_emulator();

    REQUIRE(executor->get_i_generation() == 0);
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
                REQUIRE(out_buffer_ac[c][r].i_subgeneration == 0);
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
                REQUIRE(out_buffer_ac[c][r].i_subgeneration == 0);
                REQUIRE(out_buffer_ac[c][r].status == CellStatus::Normal);
            }
        }
    }
}

TEST_CASE("TilingExecutor::run", "[TilingExecutor]") {
    TilingExecutorImpl executor(Cell::halo(), TransFunc());

    // single pass
    test_executor(&executor, tile_width, tile_height, gens_per_pass);
    // single pass, grid smaller than tile
    test_executor(&executor, tile_width / 2, tile_height / 2, gens_per_pass);
    // single pass, grid bigger than tile
    test_executor(&executor, tile_width * 2, tile_height * 2, gens_per_pass);
    // single pass, grid bigger slightly bigger than tile
    test_executor(&executor, tile_width * 1.5, tile_height * 1.5, gens_per_pass);

    // multiple passes
    test_executor(&executor, tile_width, tile_height, 2 * gens_per_pass);
    // multiple passes, grid smaller than tile
    test_executor(&executor, tile_width / 2, tile_height / 2, 2 * gens_per_pass);
    // multiple passes, grid bigger than tile
    test_executor(&executor, tile_width * 2, tile_height * 2, 2 * gens_per_pass);
    // multiple passes, grid bigger slightly bigger than tile
    test_executor(&executor, tile_width * 1.5, tile_height * 1.5, 2 * gens_per_pass);
}

TEST_CASE("MonotileExecutor::run", "[MonotileExecutor]") {
    MonotileExecutorImpl executor(Cell::halo(), TransFunc());

    // single pass
    test_executor(&executor, tile_width, tile_height, gens_per_pass);
    // single pass, grid smaller than tile
    test_executor(&executor, tile_width / 2, tile_height / 2, gens_per_pass);

    // multiple passes
    test_executor(&executor, tile_width, tile_height, 2 * gens_per_pass);
    // multiple passes, grid smaller than tile
    test_executor(&executor, tile_width / 2, tile_height / 2, 2 * gens_per_pass);
}

TEST_CASE("SimpleCPUExecutor::run", "[SimpleCPUExecutor]") {
    SimpleCPUExecutorImpl executor(Cell::halo(), TransFunc());
    test_executor(&executor, grid_width, grid_height, 32);
}

void test_snapshotting(SingleContextExecutorImpl *executor, uindex_t grid_width,
                       uindex_t grid_height, uindex_t n_generations, uindex_t delta_n_generations) {
    buffer<Cell, 2> in_buffer(range<2>(grid_width, grid_height));
    {
        auto in_buffer_ac = in_buffer.get_access<access::mode::discard_write>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                in_buffer_ac[c][r] = Cell{index_t(c), index_t(r), 0, 0, CellStatus::Normal};
            }
        }
    }

    executor->set_input(in_buffer);
    executor->set_i_generation(0);
    executor->select_emulator();

    std::vector<uindex_t> snapshotted_generations;
    auto snapshot_handler = [&](cl::sycl::buffer<Cell, 2> buffer, uindex_t i_generation) {
        auto ac = buffer.get_access<cl::sycl::access::mode::read>();
        for (uindex_t c = 0; c < grid_width; c++) {
            for (uindex_t r = 0; r < grid_height; r++) {
                REQUIRE(ac[c][r].c == c);
                REQUIRE(ac[c][r].r == r);
                REQUIRE(ac[c][r].i_generation == i_generation);
                REQUIRE(ac[c][r].i_subgeneration == 0);
                REQUIRE(ac[c][r].status == CellStatus::Normal);
            }
        }
        snapshotted_generations.push_back(i_generation);
    };

    executor->run_with_snapshots(n_generations, delta_n_generations, snapshot_handler);
    REQUIRE(executor->get_i_generation() == n_generations);

    REQUIRE(snapshotted_generations.size() ==
            n_generations / delta_n_generations +
                (n_generations % delta_n_generations == 0 ? 0 : 1));
    for (uindex_t i = 1; i < snapshotted_generations.size() - 1; i++) {
        REQUIRE(snapshotted_generations[i] == (i + 1) * delta_n_generations);
    }
    REQUIRE(snapshotted_generations.back() == n_generations);
}

TEST_CASE("TilingExecutor::run_with_snapshots", "[TilingExecutor]") {
    TilingExecutorImpl executor(Cell::halo(), TransFunc());

    // snapshot distance divides number of generations
    test_snapshotting(&executor, tile_width, tile_height, 4 * gens_per_pass, gens_per_pass);
    // snapshot distance does not divide number of generations
    test_snapshotting(&executor, tile_width, tile_height, 2.5 * gens_per_pass, gens_per_pass);
    // snapshot distance is not a multiple of pipeline length
    test_snapshotting(&executor, tile_width, tile_height, 2 * gens_per_pass, 0.5 * gens_per_pass);
}

TEST_CASE("MonotileExecutor::run_with_snapshots", "[MonotileExecutor]") {
    MonotileExecutorImpl executor(Cell::halo(), TransFunc());

    // snapshot distance divides number of generations
    test_snapshotting(&executor, tile_width, tile_height, 4 * gens_per_pass, gens_per_pass);
    // snapshot distance does not divide number of generations
    test_snapshotting(&executor, tile_width, tile_height, 2.5 * gens_per_pass, gens_per_pass);
    // snapshot distance is not a multiple of pipeline length
    test_snapshotting(&executor, tile_width, tile_height, 2 * gens_per_pass, 0.5 * gens_per_pass);
}

TEST_CASE("SimpleCPUExecutor::run_with_snapshots", "[SimpleCPUExecutor]") {
    SimpleCPUExecutorImpl executor(Cell::halo(), TransFunc());

    // snapshot distance divides number of generations
    test_snapshotting(&executor, tile_width, tile_height, 4 * gens_per_pass, gens_per_pass);
    // snapshot distance does not divide number of generations
    test_snapshotting(&executor, tile_width, tile_height, 2.5 * gens_per_pass, gens_per_pass);
    // snapshot distance is not a multiple of pipeline length
    test_snapshotting(&executor, tile_width, tile_height, 2 * gens_per_pass, 0.5 * gens_per_pass);
}