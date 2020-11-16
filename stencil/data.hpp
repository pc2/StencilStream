/*
 * Copyright © 2020 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#pragma once
#include <CL/sycl.hpp>
#include <boost/preprocessor/cat.hpp>

#define B_SIZE(radius, size) (2 * (radius) + (size))

namespace stencil
{

#ifndef STENCIL_INDEX_WIDTH
#define STENCIL_INDEX_WIDTH 64
#endif

/**
 * Integer types for indexing.
 * 
 * There is always a signed version, `index_t`, and an unsigned version, `uindex_t`. Their width is
 * defined by the `STENCIL_INDEX_WIDTH` macro. The default is 64 and can be increased to allow
 * bigger buffers or decreased to reduce the complexity and resource requirements.
 * 
 * Static asserts throughout the library ensure that the index type is wide enough. Therefore, you
 * can decrease the until you get compilation errors.
 */
typedef BOOST_PP_CAT(BOOST_PP_CAT(uint, STENCIL_INDEX_WIDTH), _t) uindex_t;
typedef BOOST_PP_CAT(BOOST_PP_CAT(int, STENCIL_INDEX_WIDTH), _t) index_t;

#ifndef STENCIL_PIPELINE_LEN
#define STENCIL_PIPELINE_LEN 1
#endif

/**
 * The length of the computation pipeline in kernel replications.
 * 
 * This value defines the number of times the stencil kernel is replicated. Bigger pipeline lengths
 * lead to increased paralellity and therefore speed, but also to increased resource use and lower clock frequency.
 * 
 * The default for this value is 1 and is set by the `STENCIL_PIPELINE_LEN` macro.
 */
const size_t pipeline_length = STENCIL_PIPELINE_LEN;

/**
 * A generic, two-dimensional index.
 */
template <typename T>
class GenericID
{
public:
    GenericID(T column, T row) : c(column), r(row) {}

    GenericID(cl::sycl::id<2> sycl_id) : c(sycl_id[0]), r(sycl_id[1]) {}

    GenericID(cl::sycl::range<2> sycl_range) : c(sycl_range[0]), r(sycl_range[1]) {}

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

/**
 * The stencil buffer.
 * 
 * The stencil buffer contains a small chunk of the grid buffer and is used by the stencil kernel to
 * calculate the next-generation cell value.
 * 
 * This implementation provides two ways to index the stencil: With an `ID` and a `UID`. Since `ID` is signed, the column and row axies are within the range of [-radius : radius].
 * Therefore, (0,0) points to the middle of the stencil. This is the value that is going to be replaced. 
 * `UID` is unsigned and the column and row axies are within the range of [0 : 2*radius + 1). Therefore, (0,0) points to the upper-left corner of the stencil.
 */
template <typename T, index_t radius>
class Stencil2D
{
    static_assert(B_SIZE(radius, 1) < std::numeric_limits<uindex_t>::max());
    static_assert(B_SIZE(radius, 1) >= 3);

public:
    Stencil2D() : internal() {}

    T const &operator[](ID id) const { return internal[id.c + radius][id.r + radius]; }

    T &operator[](ID id) { return internal[id.c + radius][id.r + radius]; }

    T const &operator[](UID id) const { return internal[id.c][id.r]; }

    T &operator[](UID id) { return internal[id.c][id.r]; }

private:
    T internal[B_SIZE(radius, 1)][B_SIZE(radius, 1)];
};

/**
 * All parameters for the stencil kernel.
 */
struct Stencil2DInfo
{
    /**
     * The position of the central cell in the global grid. 
     * 
     * This is the position of the cell the stencil kernel has to calculate and the position of the central cell of the stencil buffer.
     */
    UID center_cell_id = UID(0, 0);
    /**
     * The present generation of the central cell of the stencil buffer.
     * 
     * This number +1 is the generation of the cell the stencil kernel calculates.
     */
    index_t cell_generation = 0;
    /**
     * True if the invocation of the stencil kernel may have side effects like sending something through a pipe.
     * 
     * If the number of generations demanded by the user isn't a multiple of the pipeline length,
     * the stencil kernel will executed a couple times to often in order to bring number of calculated
     * generations to a multiple of the pipeline length. The cells produced by these "padding invocations"
     * are discarded and they work on possible incorrect data. Therefore, these padding invocations
     * may not have side effects since the library can not discard these side effects too.
     * 
     * The first invocation of the stencil kenrel in the pipeline is an exception to this rule since
     * it's results are never discarded due to it's position (If it's not needed, it isn't executed
     * since the whole pass isn't needed).
     * 
     * Therefore, if you want to send some information to another kernel via a pipe, do it exactly when
     * `may_have_sideeffects` is true.
     */
    bool may_have_sideeffects;
    /**
     * The position of the stencil kernel replication.
     * 
     * This can be used to execute certain operations only once per loop execution, for example sending
     * an intermediate value to another kernel.
     */
    uindex_t pipeline_position = 0;
};

using sync_buffer = cl::sycl::buffer<uint8_t, 1>;
} // namespace stencil