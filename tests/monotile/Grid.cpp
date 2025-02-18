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
#include "../GridTest.hpp"
#include <StencilStream/monotile/Grid.hpp>

using namespace stencil;
using namespace stencil::monotile;
using namespace sycl;
using namespace std;

// Assert that the monotile grid fulfills the grid concept.
static_assert(concepts::Grid<Grid<sycl::id<2>>, sycl::id<2>>);

constexpr std::size_t max_grid_height = 64;
constexpr std::size_t max_grid_width = 32;

TEST_CASE("monotile::Grid::Grid", "[monotile::Grid]") {
    grid_test::test_constructors<Grid<sycl::id<2>>>(max_grid_height, max_grid_width);
}

TEST_CASE("monotile::Grid::copy_from_buffer", "[monotile::Grid]") {
    grid_test::test_copy_from_buffer<Grid<sycl::id<2>>>(max_grid_height, max_grid_width);
}

TEST_CASE("monotile::Grid::copy_to_buffer", "[monotile::Grid]") {
    grid_test::test_copy_to_buffer<Grid<sycl::id<2>>>(max_grid_height, max_grid_width);
}

TEST_CASE("monotile::Grid::make_similar", "[monotile::Grid]") {
    grid_test::test_make_similar<Grid<sycl::id<2>>>(max_grid_height, max_grid_width);
}

template <std::size_t spatial_parallelism>
void test_monotile_grid_submit_read(std::size_t grid_height, std::size_t grid_width) {
    using TestGrid = monotile::Grid<sycl::id<2>, spatial_parallelism>;

    TestGrid in_grid(grid_height, grid_width);
    {
        using GridAccessor = TestGrid::template GridAccessor<access::mode::read_write>;
        GridAccessor in_grid_ac(in_grid);
        for (std::size_t r = 0; r < grid_height; r++) {
            for (std::size_t c = 0; c < grid_width; c++) {
                in_grid_ac[r][c] = sycl::id<2>(r, c);
            }
        }
    }

    sycl::queue queue;

    using in_pipe = sycl::pipe<class monotile_grid_submit_read_test_id,
                               std::array<sycl::id<2>, spatial_parallelism>>;
    in_grid.template submit_read<in_pipe, max_grid_height, max_grid_width>(queue);

    buffer<sycl::id<2>, 2> out_buffer = range<2>(grid_height, grid_width);
    queue.submit([&](handler &cgh) {
        accessor out_ac(out_buffer, cgh, sycl::write_only);

        cgh.single_task([=]() {
            for (std::size_t r = 0; r < grid_height; r++) {
                for (std::size_t c = 0; c < grid_width; c += spatial_parallelism) {
                    std::array<sycl::id<2>, spatial_parallelism> vector = in_pipe::read();
#pragma unroll
                    for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                        if (c + i_cell < grid_width) {
                            out_ac[r][c + i_cell] = vector[i_cell];
                        }
                    }
                }
            }
        });
    });

    host_accessor out_buffer_ac(out_buffer, sycl::read_only);
    for (std::size_t r = 0; r < grid_height; r++) {
        for (std::size_t c = 0; c < grid_width; c += spatial_parallelism) {
            REQUIRE(out_buffer_ac[r][c] == sycl::id<2>(r, c));
        }
    }
}

TEST_CASE("monotile::Grid::submit_read", "[monotile::Grid]") {
    test_monotile_grid_submit_read<1>(max_grid_height, max_grid_width);
    test_monotile_grid_submit_read<2>(max_grid_height, max_grid_width);
    test_monotile_grid_submit_read<4>(max_grid_height, max_grid_width);

    test_monotile_grid_submit_read<1>(max_grid_height - 1, max_grid_width);
    test_monotile_grid_submit_read<2>(max_grid_height - 1, max_grid_width);
    test_monotile_grid_submit_read<4>(max_grid_height - 1, max_grid_width);

    test_monotile_grid_submit_read<1>(max_grid_height, max_grid_width - 1);
    test_monotile_grid_submit_read<2>(max_grid_height, max_grid_width - 1);
    test_monotile_grid_submit_read<4>(max_grid_height, max_grid_width - 1);

    test_monotile_grid_submit_read<1>(max_grid_height - 1, max_grid_width - 1);
    test_monotile_grid_submit_read<2>(max_grid_height - 1, max_grid_width - 1);
    test_monotile_grid_submit_read<4>(max_grid_height - 1, max_grid_width - 1);
}

template <std::size_t spatial_parallelism>
void test_monotile_grid_submit_write(std::size_t grid_height, std::size_t grid_width) {
    using TestGrid = monotile::Grid<sycl::id<2>, spatial_parallelism>;
    using out_pipe = sycl::pipe<class monotile_grid_submit_write_test_id,
                                std::array<sycl::id<2>, spatial_parallelism>>;

    sycl::queue queue;
    queue.submit([&](sycl::handler &cgh) {
        cgh.single_task([=]() {
            for (std::size_t r = 0; r < grid_height; r++) {
                for (std::size_t c = 0; c < grid_width; c += spatial_parallelism) {
                    std::array<sycl::id<2>, spatial_parallelism> vector;
#pragma unroll
                    for (std::size_t i_cell = 0; i_cell < spatial_parallelism; i_cell++) {
                        vector[i_cell] = sycl::id<2>(r, c + i_cell);
                    }
                    out_pipe::write(vector);
                }
            }
        });
    });

    TestGrid grid(grid_height, grid_width);
    grid.template submit_write<out_pipe, max_grid_height, max_grid_width>(queue);

    using GridAccessor = TestGrid::template GridAccessor<access::mode::read_write>;
    GridAccessor out_ac(grid);
    for (std::size_t r = 0; r < grid_height; r++) {
        for (std::size_t c = 0; c < grid_width; c++) {
            REQUIRE(out_ac[r][c] == sycl::id<2>(r, c));
        }
    }
}

TEST_CASE("monotile::Grid::submit_write", "[monotile::Grid]") {
    test_monotile_grid_submit_write<1>(max_grid_height, max_grid_width);
    test_monotile_grid_submit_write<2>(max_grid_height, max_grid_width);
    test_monotile_grid_submit_write<4>(max_grid_height, max_grid_width);

    test_monotile_grid_submit_write<1>(max_grid_height - 1, max_grid_width);
    test_monotile_grid_submit_write<2>(max_grid_height - 1, max_grid_width);
    test_monotile_grid_submit_write<4>(max_grid_height - 1, max_grid_width);

    test_monotile_grid_submit_write<1>(max_grid_height, max_grid_width - 1);
    test_monotile_grid_submit_write<2>(max_grid_height, max_grid_width - 1);
    test_monotile_grid_submit_write<4>(max_grid_height, max_grid_width - 1);

    test_monotile_grid_submit_write<1>(max_grid_height - 1, max_grid_width - 1);
    test_monotile_grid_submit_write<2>(max_grid_height - 1, max_grid_width - 1);
    test_monotile_grid_submit_write<4>(max_grid_height - 1, max_grid_width - 1);
}