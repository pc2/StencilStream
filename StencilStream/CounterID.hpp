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
 * \brief A subclass of the `GenericID<T>` with two-dimensional index counting.
 * 
 * It additionally requires bounds for the column and row counters and provides increment and decrement
 * operators. The row counter is always the fastest: For example, the increment operators first increase
 * the row counter and when it reaches the row bound, it is reset to zero and the column counter is
 * increased. When the column counter reaches it's bound, it is also reset to zero. One can therefore
 * use this counter to safely iterate over a two-dimensional grid even with multiple passes.
 */
template <typename T>
class CounterID : public GenericID<T>
{
public:
    /**
     * \brief Create a new counter with uninitialized column and row counters.
     * 
     * The column and row bounds may not be uninitialized and are therefore required.
     */
    CounterID(T column_bound, T row_bound) : GenericID<T>(), c_bound(column_bound), r_bound(row_bound) {}

    /**
     * \brief Create a new counter with the given column/row counters and bounds.
     */
    CounterID(T column, T row, T column_bound, T row_bound) : GenericID<T>(column, row), c_bound(column_bound), r_bound(row_bound) {}

    /**
     * \brief Create a new counter from the given SYCL id and bounds.
     */
    CounterID(cl::sycl::id<2> sycl_id, T column_bound, T row_bound) : GenericID<T>(sycl_id), c_bound(column_bound), r_bound(row_bound) {}

    /**
     * \brief Increase the counters.
     */
    CounterID &operator++()
    {
        if (this->r == r_bound - 1)
        {
            this->r = 0;
            if (this->c == c_bound - 1)
            {
                this->c = 0;
            }
            else
            {
                this->c++;
            }
        }
        else
        {
            this->r++;
        }
        return *this;
    }

    /**
     * \brief Decrease the counters.
     */
    CounterID &operator--()
    {
        if (this->r == 0)
        {
            this->r = r_bound - 1;
            if (this->c == 0)
            {
                this->c = c_bound - 1;
            }
            else
            {
                this->c--;
            }
        }
        else
        {
            this->r--;
        }
        return *this;
    }

    /**
     * \brief Increase the counters.
     */
    CounterID operator++(int)
    {
        CounterID copy = *this;
        this->operator++();
        return copy;
    }

    /**
     * \brief Decrease the counters.
     */
    CounterID operator--(int)
    {
        CounterID copy = *this;
        this->operator--();
        return copy;
    }

private:
    T c_bound;
    T r_bound;
};
} // namespace stencil