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
#include "HostPipe.hpp"
#include <catch2/catch_all.hpp>

using namespace stencil;

TEST_CASE("HostPipe (normal)", "[HostPipe]") {
    using MyPipe = HostPipe<class MyPipeID, uindex_t>;
    for (uindex_t i = 0; i < 256; i++) {
        MyPipe::write(i);
    }
    for (uindex_t i = 0; i < 256; i++) {
        REQUIRE(MyPipe::read() == i);
    }
    REQUIRE(MyPipe::empty());
}

TEST_CASE("HostPipe (separated pipes)", "[HostPipe]") {
    using PipeA = HostPipe<class PipeAID, index_t>;
    using PipeB = HostPipe<class PipeBID, index_t>;

    for (index_t i = 0; i < 256; i++) {
        PipeA::write(i);
        PipeB::write(-1 * i);
    }

    for (index_t i = 0; i < 256; i++) {
        REQUIRE(PipeA::read() == i);
        REQUIRE(PipeB::read() == -1 * i);
    }

    REQUIRE(PipeA::empty());
    REQUIRE(PipeB::empty());
}

TEST_CASE("HostPipe (continous read/write)", "[HostPipe]") {
    using PipeC = HostPipe<class PipeCID, uindex_t>;

    PipeC::write(0);
    PipeC::write(1);

    for (uindex_t i = 0; i < 256; i++) {
        REQUIRE(PipeC::read() == i);
        PipeC::write(i + 2);
    }

    REQUIRE(!PipeC::empty());
}
