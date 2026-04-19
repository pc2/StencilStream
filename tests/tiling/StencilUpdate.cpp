/*
 * Copyright © 2020-2026 Paderborn Center for Parallel Computing, Paderborn
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
#include "../HostPipe.hpp"
#include "../StencilUpdateTest.hpp"
#include "../TransFuncs.hpp"

#include <StencilStream/tiling/StencilUpdate.hpp>
#include <catch2/catch_all.hpp>

using namespace sycl;
using namespace stencil;
using namespace stencil::tiling;

TEST_CASE("tiling::StencilUpdate", "[tiling::StencilUpdate]") {
    constexpr std::size_t temporal_parallelism = 4;
    constexpr std::size_t tile_height = 128;
    constexpr std::size_t tile_width = 128;

    using StencilUpdateVect1Impl =
        StencilUpdate<FPGATransFunc<1>, temporal_parallelism, 1, tile_height, tile_width, 1>;
    using GridVect1Impl = Grid<FPGATransFunc<1>::Cell, 1>;

    using StencilUpdateVect2Impl =
        StencilUpdate<FPGATransFunc<1>, temporal_parallelism, 2, tile_height, tile_width, 1>;
    using GridVect2Impl = Grid<FPGATransFunc<1>::Cell, 2>;

    using StencilUpdateVect4Impl =
        StencilUpdate<FPGATransFunc<1>, temporal_parallelism, 4, tile_height, tile_width, 1>;
    using StencilUpdateVect4SplitImpl =
        StencilUpdate<FPGATransFunc<1>, temporal_parallelism, 4, tile_height, tile_width, 2>;
    using GridVect4Impl = Grid<FPGATransFunc<1>::Cell, 4>;

    static_assert(concepts::StencilUpdate<StencilUpdateVect1Impl, FPGATransFunc<1>, GridVect1Impl>);
    static_assert(concepts::StencilUpdate<StencilUpdateVect2Impl, FPGATransFunc<1>, GridVect2Impl>);
    static_assert(concepts::StencilUpdate<StencilUpdateVect4Impl, FPGATransFunc<1>, GridVect4Impl>);

    for (std::size_t i_grid_height = 0; i_grid_height < 3; i_grid_height++) {
        for (std::size_t i_grid_width = 0; i_grid_width < 3; i_grid_width++) {
            std::size_t grid_height = (1 + i_grid_height) * (tile_height / 2);
            std::size_t grid_width = (1 + i_grid_width) * (tile_width / 2);

            test_stencil_update<GridVect1Impl, StencilUpdateVect1Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect1Impl, StencilUpdateVect1Impl>(grid_height, grid_width, 1,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect1Impl, StencilUpdateVect1Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism + 1);
            test_stencil_update<GridVect1Impl, StencilUpdateVect1Impl>(grid_height - 1, grid_width,
                                                                       0, temporal_parallelism + 1);
            test_stencil_update<GridVect1Impl, StencilUpdateVect1Impl>(grid_height, grid_width - 1,
                                                                       0, temporal_parallelism + 1);

            test_stencil_update<GridVect2Impl, StencilUpdateVect2Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect2Impl, StencilUpdateVect2Impl>(grid_height, grid_width, 1,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect2Impl, StencilUpdateVect2Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism + 1);
            test_stencil_update<GridVect2Impl, StencilUpdateVect2Impl>(grid_height - 1, grid_width,
                                                                       0, temporal_parallelism + 1);
            test_stencil_update<GridVect2Impl, StencilUpdateVect2Impl>(grid_height, grid_width - 1,
                                                                       0, temporal_parallelism + 1);

            test_stencil_update<GridVect4Impl, StencilUpdateVect4Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4Impl>(grid_height, grid_width, 1,
                                                                       temporal_parallelism);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4Impl>(grid_height, grid_width, 0,
                                                                       temporal_parallelism + 1);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4Impl>(grid_height - 1, grid_width,
                                                                       0, temporal_parallelism + 1);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4Impl>(grid_height, grid_width - 1,
                                                                       0, temporal_parallelism + 1);

            test_stencil_update<GridVect4Impl, StencilUpdateVect4SplitImpl>(
                grid_height, grid_width, 0, temporal_parallelism);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4SplitImpl>(
                grid_height, grid_width, 1, temporal_parallelism);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4SplitImpl>(
                grid_height, grid_width, 0, temporal_parallelism + 1);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4SplitImpl>(
                grid_height - 1, grid_width, 0, temporal_parallelism + 1);
            test_stencil_update<GridVect4Impl, StencilUpdateVect4SplitImpl>(
                grid_height, grid_width - 1, 0, temporal_parallelism + 1);
        }
    }
}