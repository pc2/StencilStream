#pragma once
#include <sycl/sycl.hpp>

// This accessor is shifted by 1. It is used for the padded access idea.

template <typename Cell, typename Accessor, sycl::access::mode access_mode,
          std::size_t current_subdim = 0>
class AccessorSubscript {
  public:
    static constexpr std::size_t dimensions = Accessor::dimensions;

    AccessorSubscript(Accessor &ac, std::size_t i)
        requires(current_subdim == 0)
        : ac(ac), id_prefix() {
        id_prefix[current_subdim] = i + 1;
    }

    AccessorSubscript(Accessor &ac, sycl::id<dimensions> id_prefix, std::size_t i)
        : ac(ac), id_prefix(id_prefix) {
        id_prefix[current_subdim] = i + 1;
    }

    AccessorSubscript<Cell, Accessor, access_mode, current_subdim + 1> operator[](std::size_t i)
        requires(current_subdim < dimensions - 2)
    {
        return AccessorSubscript(ac, id_prefix, i + 1);
    }

    Cell const &operator[](std::size_t i)
        requires(current_subdim == dimensions - 2 && access_mode == sycl::access::mode::read)
    {
        sycl::id<dimensions> id = id_prefix;
        id[current_subdim + 1] = i + 1;
        return ac[id];
    }

    Cell &operator[](std::size_t i)
        requires(current_subdim == dimensions - 2 && access_mode != sycl::access::mode::read)
    {
        sycl::id<dimensions> id = id_prefix;
        id[current_subdim + 1] = i + 1;
        return ac[id];
    }

  private:
    Accessor &ac;
    sycl::id<dimensions> id_prefix;
};
