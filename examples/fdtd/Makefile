# Copyright © 2020-2021Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
CC = dpcpp
STENCIL_STREAM_PATH = ./

ARGS = -fintelfpga -std=c++17 -DSTENCIL_INDEX_WIDTH=64 -DFDTD_BURST_SIZE=1024 -I$(STENCIL_STREAM_PATH) -O3
HWARGS =

ifdef EBROOTGCC
	ARGS += --gcc-toolchain=$(EBROOTGCC)
endif

ifdef AOCL_BOARD_PACKAGE_ROOT
	HWARGS += -Xsv -Xsprofile -Xsboard=$(FPGA_BOARD_NAME) -Xsboard-package=$(AOCL_BOARD_PACKAGE_ROOT)
endif

EMU =
HW = -DHARDWARE -Xshardware 
MONOTILE = -DMONOTILE
TILING =

SOURCES = $(wildcard src/*.cpp)
RESOURCES = $(wildcard src/*.hpp) $(wildcard StencilStream/*) Makefile

all: fdtd_mono_emu fdtd_tiling_emu fdtd_mono_hw fdtd_tiling_hw

fdtd_mono_emu: $(SOURCES) $(RESOURCES)
	$(CC) $(ARGS) $(MONOTILE) $(EMU) $(SOURCES) -o fdtd_mono_emu

fdtd_tiling_emu: $(SOURCES) $(RESOURCES)
	$(CC) $(ARGS) $(TILING) $(EMU) $(SOURCES) -o fdtd_tiling_emu

fdtd_mono_hw: $(SOURCES) $(RESOURCES)
	$(CC) $(ARGS) $(HWARGS) $(MONOTILE) $(HW) $(SOURCES) -o fdtd_mono_hw

fdtd_mono_hw.report.tar.gz: $(SOURCES) $(RESOURCES)
	rm -f fdtd_mono_hw
	$(CC) $(ARGS) $(HWARGS) $(MONOTILE) $(HW) -fsycl-link $(SOURCES) -o fdtd_mono_hw
	tar -caf fdtd_mono_hw.report.tar.gz fdtd_mono_hw.prj/reports

fdtd_tiling_hw: $(SOURCES) $(RESOURCES)
	$(CC) $(ARGS) $(HWARGS) $(TILING) $(HW) $(SOURCES) -o fdtd_tiling_hw

fdtd_tiling_hw.report.tar.gz: $(SOURCES) $(RESOURCES)
	rm -f fdtd_tiling_hw
	$(CC) $(ARGS) $(HWARGS) $(TILING) $(HW) -fsycl-link $(SOURCES) -o fdtd_tiling_hw
	tar -caf fdtd_tiling_hw.report.tar.gz fdtd_tiling_hw.prj/reports

clean:
	git clean -dXf
