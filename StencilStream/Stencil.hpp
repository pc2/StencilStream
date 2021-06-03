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
#include "GenericID.hpp"
#include "Index.hpp"

namespace stencil
{

/**
 * The stencil buffer.
 * 
 * The stencil buffer contains a small chunk of the grid buffer and is used by the transition function to
 * calculate the next-generation cell value.
 * 
 * This implementation provides two ways to index the stencil: With an `ID` and a `UID`. Since `ID` is signed, the column and row axies are within the range of [-radius : radius].
 * Therefore, (0,0) points to the middle of the stencil. This is the value that is going to be replaced. 
 * `UID` is unsigned and the column and row axies are within the range of [0 : 2*radius + 1). Therefore, (0,0) points to the upper-left corner of the stencil.
 */
template <typename T, uindex_t radius>
class Stencil
{
public:
    static constexpr uindex_t diameter()
    {
        return 2 * radius + 1;
    }

    static_assert(diameter() < std::numeric_limits<uindex_t>::max());
    static_assert(diameter() >= 3);

    Stencil(ID id, uindex_t generation, uindex_t stage) : id(id), generation(generation), stage(stage), internal() {}

    Stencil(ID id, uindex_t generation, uindex_t stage, T raw[diameter()][diameter()]) : id(id), generation(generation), stage(stage), internal()
    {
#pragma unroll
        for (uindex_t c = 0; c < diameter(); c++)
        {
#pragma unroll
            for (uindex_t r = 0; r < diameter(); r++)
            {
                internal[c][r] = raw[c][r];
            }
        }
    }

    T const &operator[](ID id) const { return internal[id.c + radius][id.r + radius]; }

    T &operator[](ID id) { return internal[id.c + radius][id.r + radius]; }

    T const &operator[](UID id) const { return internal[id.c][id.r]; }

    T &operator[](UID id) { return internal[id.c][id.r]; }

    /**
     * The position of the central cell in the global grid. 
     * 
     * This is the position of the cell the transition has to calculate and the position of the central cell of the stencil buffer.
     */
    const ID id;
    /**
     * The present generation of the central cell of the stencil buffer.
     * 
     * This number +1 is the generation of the cell the transition function calculates.
     */
    const uindex_t generation;
    /**
     * The index of the pipeline stage that calls the transition function.
     * 
     * The stage index is the offset added to the generation index of the input tile to get the generation
     * index of this stencil. It is hardcoded in the design and under the assumption that the pipeline
     * was always fully executed, it equals to `generation % pipeline_length`.
     */
    const uindex_t stage;

private:
    T internal[diameter()][diameter()];
};

} // namespace stencil