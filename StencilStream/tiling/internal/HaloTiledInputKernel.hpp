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
#include "../../internal/Helpers.hpp"
#include "../Grid.hpp"
#include <bit>
#include <exception>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

namespace stencil {
namespace tiling {
namespace internal {

template <typename Cell, std::size_t spatial_parallelism, typename out_pipe,
          std::size_t max_tile_height, std::size_t max_tile_width, std::size_t halo_height,
          std::size_t halo_width>
class HaloTiledInputKernel {
  public:
    static constexpr std::size_t max_vect_tile_width = max_tile_width / spatial_parallelism;
    static_assert(max_tile_width % spatial_parallelism == 0);
    static constexpr std::size_t vect_halo_width = halo_width / spatial_parallelism;
    static_assert(halo_width % spatial_parallelism == 0);

    using GridImpl = Grid<Cell, spatial_parallelism>;
    using CellVector = typename GridImpl::CellVector;
    using Accessor =
        sycl::accessor<CellVector, 2, sycl::access::mode::read, sycl::access::target::device>;
    using uindex_r_t = ac_int<std::bit_width(max_tile_height + 2 * halo_height), false>;
    using uindex_vect_c_t =
        ac_int<std::bit_width(max_vect_tile_width + 2 * vect_halo_width), false>;

    HaloTiledInputKernel(GridImpl grid, sycl::handler &cgh, sycl::id<2> tile_id, Cell halo_value)
        : accessor(grid.get_internal().get_access(cgh, sycl::read_only)),
          grid_height(grid.get_grid_height()), grid_width(grid.get_grid_width(false)), vect_grid_width(grid.get_grid_width(true)), vect_tile_offset(0, 0), tile_height(0),
          vect_tile_width(0), halo_value(halo_value) {
        sycl::range<2> max_tile_range(max_tile_height, max_tile_width);
        sycl::range<2> tile_id_range = grid.get_tile_id_range(max_tile_range);

        vect_tile_offset = grid.get_tile_offset(tile_id, max_tile_range, true);

        sycl::range<2> vect_tile_range = grid.get_tile_range(tile_id, max_tile_range, true);
        tile_height = vect_tile_range[0];
        vect_tile_width = vect_tile_range[1];
    }

    void operator()() const {
        uindex_r_t haloed_tile_height = tile_height + uindex_r_t(2 * halo_height);
        uindex_vect_c_t haloed_vect_tile_width =
            vect_tile_width + uindex_vect_c_t(2 * vect_halo_width);

        [[intel::loop_coalesce(2)]]
        for (uindex_r_t local_r = 0; local_r < haloed_tile_height; local_r++) {
            for (uindex_vect_c_t local_vect_c = 0; local_vect_c < haloed_vect_tile_width;
                 local_vect_c++) {
                std::size_t r = (local_r - halo_height).to_ulong() + vect_tile_offset[0];
                std::size_t vect_c =
                    (local_vect_c - vect_halo_width).to_ulong() + vect_tile_offset[1];

                bool is_grid_halo;
                if (local_r < uindex_r_t(halo_height)) {
                    is_grid_halo = vect_tile_offset[0] < halo_height;
                } else {
                    is_grid_halo = r >= grid_height;
                }
                if (local_vect_c < uindex_vect_c_t(vect_halo_width)) {
                    is_grid_halo |= vect_tile_offset[1] < vect_halo_width;
                } else {
                    is_grid_halo |= vect_c >= vect_grid_width;
                }

                CellVector cell_vector;
                if (!is_grid_halo) {
                    cell_vector = accessor[r][vect_c];
                }

#pragma unroll
                for (std::size_t cell_i = 0; cell_i < spatial_parallelism; cell_i++) {
                    std::size_t c = vect_c * spatial_parallelism + cell_i;
                    if (is_grid_halo || (local_vect_c >= vect_halo_width && c >= grid_width)) {
                        cell_vector.value[cell_i] = halo_value;
                    }
                }

                out_pipe::write(cell_vector);
            }
        }
    }

  private:
    Accessor accessor;
    std::size_t grid_height;
    std::size_t grid_width;
    std::size_t vect_grid_width;
    sycl::id<2> vect_tile_offset;
    uindex_r_t tile_height;
    uindex_vect_c_t vect_tile_width;
    Cell halo_value;
};

} // namespace internal
} // namespace tiling
} // namespace stencil