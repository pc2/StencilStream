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
#include <numeric>

namespace stencil {

/**
 * \brief Compute the number of words necessary to store a given number of cells.
 *
 * Some backends store cells in groups called words. Each word contains a certain number of cells,
 * and this function computes how many words are needed to store a certain number of cells. This is
 * the total number of cells divided by the number of cells in a word, plus one additional word if
 * the word length doesn't divide the total number of cells.
 */
inline constexpr uindex_t n_cells_to_n_words(uindex_t n_cells, uindex_t word_length) {
    return n_cells / word_length + (n_cells % word_length == 0 ? 0 : 1);
}

/**
 * \brief A container with padding to the next power of two.
 *
 * Wrapping a type in this template ensures that the resulting size is a power of two.
 *
 * \tparam T The contained type.
 */
template <typename T> struct Padded {
    T value;
} __attribute__((aligned(std::bit_ceil(sizeof(T)))));

} // namespace stencil