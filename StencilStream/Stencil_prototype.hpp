#pragma once
#include <bit>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/id.hpp>
#include <sycl/range.hpp>
#include <variant>

template <typename Cell, std::size_t stencil_radius>
    requires std::semiregular<Cell> && (stencil_radius >= 1)
class Stencil {
  public:
    static constexpr std::size_t diameter = 2 * stencil_radius + 1;
    static_assert(diameter <= std::numeric_limits<int>::max());

    Stencil(sycl::id<2> id, sycl::range<2> grid_range, std::size_t iteration,
            std::size_t subiteration)
        : id(id), iteration(iteration), subiteration(subiteration), grid_range(grid_range),
          internal() {}

    Stencil(sycl::id<2> id, sycl::range<2> grid_range, std::size_t iteration,
            std::size_t subiteration, Cell raw[diameter][diameter])
        : id(id), iteration(iteration), subiteration(subiteration), grid_range(grid_range),
          internal() {
#pragma unroll
        for (std::size_t r = 0; r < diameter; r++) {
#pragma unroll
            for (std::size_t c = 0; c < diameter; c++) {
                internal[r][c] = raw[r][c];
            }
        }
    }

    template <std::signed_integral index_t>
        requires(stencil_radius <= std::numeric_limits<index_t>::max())
    class StencilSubscript {
      public:
        StencilSubscript(Stencil const &stencil, index_t r) : stencil(stencil), r(r) {}

        Cell const &operator[](index_t c) const {
            return stencil[sycl::id<2>(r + stencil_radius, c + stencil_radius)];
        }

      private:
        Stencil const &stencil;
        index_t r;
    };

    template <std::signed_integral index_t>
    StencilSubscript<index_t> operator[](index_t r) const
        requires(stencil_radius <= std::numeric_limits<index_t>::max())
    {
        return StencilSubscript<index_t>(*this, r);
    }

    Cell const &operator[](sycl::id<2> id) const { return internal[id[0]][id[1]]; }

    Cell &operator[](sycl::id<2> id) { return internal[id[0]][id[1]]; }

    const sycl::id<2> id;

    const std::size_t iteration;

    const std::size_t subiteration;

    const sycl::range<2> grid_range;

  private:
    Cell internal[diameter][diameter];
};