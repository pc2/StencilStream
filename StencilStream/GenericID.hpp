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
#include "Index.hpp"
#include <CL/sycl.hpp>

namespace stencil {

/**
 * \brief A generic, two-dimensional index.
 *
 * \tparam The index type. It can be anything as long as it can be constructed from a dimension of
 * `sycl::id` and tested for equality.
 */
template <typename T> class GenericID {
  public:
    /**
     * \brief Create a new index with undefined contents.
     */
    GenericID() : c(), r() {}

    /**
     * \brief Create a new index with the given column and row indices.
     */
    GenericID(T column, T row) : c(column), r(row) {}

    /**
     * \brief Convert the SYCL ID.
     */
    GenericID(sycl::id<2> sycl_id) : c(sycl_id[0]), r(sycl_id[1]) {}

    /**
     * \brief Convert the SYCl range.
     */
    GenericID(sycl::range<2> sycl_range) : c(sycl_range[0]), r(sycl_range[1]) {}

    /**
     * \brief Test if the other generic ID has equivalent coordinates to this ID.
     */
    bool operator==(GenericID const &other) const {
        return this->c == other.c && this->r == other.r;
    }

    /**
     * \brief The column index.
     */
    T c;

    /**
     * \brief The row index.
     */
    T r;
};

/**
 * \brief A signed, two-dimensional index.
 */
typedef GenericID<index_t> ID;

/**
 * \brief An unsigned, two-dimensional index.
 */
typedef GenericID<uindex_t> UID;

} // namespace stencil