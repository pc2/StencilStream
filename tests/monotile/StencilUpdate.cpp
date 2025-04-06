/*
 * Copyright © 2020-2024 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn
 * University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "../StencilUpdateTest.hpp"
#include "../TransFuncs.hpp"
#include <StencilStream/monotile/StencilUpdate.hpp>
#include <catch2/catch_all.hpp>

using namespace sycl;
using namespace stencil;
using namespace stencil::monotile;

constexpr std::size_t stencil_radius = 2;
constexpr std::size_t tile_height = 64;
constexpr std::size_t tile_width = 32;
constexpr std::size_t temporal_parallelism = 4;
using TransFunc = HostTransFunc<stencil_radius>;

template <typename TDVStrategy, std::size_t spatial_parallelism, size_t n_kernels>
void test_monotile_update() {
    using StencilUpdateImpl =
        StencilUpdate<FPGATransFunc<1>, temporal_parallelism, spatial_parallelism, tile_height,
                      tile_width, n_kernels, TDVStrategy>;
    using GridImpl = StencilUpdateImpl::GridImpl;
    static_assert(concepts::StencilUpdate<StencilUpdateImpl, FPGATransFunc<1>, GridImpl>);

    for (std::size_t grid_height = tile_height / 2; grid_height <= tile_height; grid_height *= 2) {
        for (std::size_t grid_width = tile_width / 2; grid_width <= tile_width; grid_width *= 2) {
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_height, grid_width, 0,
                                                             temporal_parallelism);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_height, grid_width, 1,
                                                             temporal_parallelism);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_height, grid_width, 0,
                                                             temporal_parallelism + 1);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_height - 1, grid_width, 0,
                                                             temporal_parallelism + 1);
            test_stencil_update<GridImpl, StencilUpdateImpl>(grid_height, grid_width - 1, 0,
                                                             temporal_parallelism + 1);
        }
    }
}

TEST_CASE("monotile::StencilUpdate", "[monotile::StencilUpdate]") {
    test_monotile_update<tdv::single_pass::InlineStrategy, 1, 1>();
    test_monotile_update<tdv::single_pass::PrecomputeOnDeviceStrategy, 1, 1>();
    test_monotile_update<tdv::single_pass::PrecomputeOnHostStrategy, 1, 1>();

    test_monotile_update<tdv::single_pass::InlineStrategy, 4, 1>();
    test_monotile_update<tdv::single_pass::PrecomputeOnDeviceStrategy, 4, 1>();
    test_monotile_update<tdv::single_pass::PrecomputeOnHostStrategy, 4, 1>();

    test_monotile_update<tdv::single_pass::InlineStrategy, 4, 2>();
    test_monotile_update<tdv::single_pass::PrecomputeOnDeviceStrategy, 4, 2>();
    test_monotile_update<tdv::single_pass::PrecomputeOnHostStrategy, 4, 2>();

    test_monotile_update<tdv::single_pass::InlineStrategy, 4, 3>();
    test_monotile_update<tdv::single_pass::PrecomputeOnDeviceStrategy, 4, 3>();
    test_monotile_update<tdv::single_pass::PrecomputeOnHostStrategy, 4, 3>();
}