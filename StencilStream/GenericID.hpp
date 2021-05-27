/*
 * Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include "Index.hpp"
#include <CL/sycl/id.hpp>

namespace stencil
{

/**
 * A generic, two-dimensional index.
 */
template <typename T>
class GenericID
{
public:
    GenericID() : c(), r() {}

    GenericID(T column, T row) : c(column), r(row) {}

    GenericID(cl::sycl::id<2> sycl_id) : c(sycl_id[0]), r(sycl_id[1]) {}

    GenericID(cl::sycl::range<2> sycl_range) : c(sycl_range[0]), r(sycl_range[1]) {}

    bool operator==(GenericID const &other) const
    {
        return this->c == other.c && this->r == other.r;
    }

    T c, r;
};

/**
 * A signed, two-dimensional index.
 */
typedef GenericID<index_t> ID;

/**
 * An unsigned, two-dimensional index.
 */
typedef GenericID<uindex_t> UID;

} // namespace stencil