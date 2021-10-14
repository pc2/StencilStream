# Copyright © 2020-2021Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
CC = dpcpp
STENCIL_STREAM_PATH = ./

ARGS = -fintelfpga -std=c++17 -I$(STENCIL_STREAM_PATH) -O3

ifdef EBROOTGCC
	ARGS += --gcc-toolchain=$(EBROOTGCC)
endif

EMU =
HW = -DHARDWARE -Xshardware -Xsv -Xsprofile 
MONOTILE = -DMONOTILE
TILING =

ifdef AOCL_BOARD_PACKAGE_ROOT
	HW += -Xsboard=$(FPGA_BOARD_NAME) -Xsboard-package=$(AOCL_BOARD_PACKAGE_ROOT)
endif

SOURCES = hotspot.cpp
RESOURCES = $(wildcard StencilStream/*) Makefile

all: hotspot_mono_emu hotspot_tiling_emu hotspot_rodinia hotspot_mono_hw hotspot_tiling_hw

hotspot_mono_emu: $(SOURCES) $(RESOURCES)
	$(CC) $(ARGS) $(MONOTILE) $(EMU) $(SOURCES) -o hotspot_mono_emu

hotspot_tiling_emu: $(SOURCES) $(RESOURCES)
	$(CC) $(ARGS) $(TILING) $(EMU) $(SOURCES) -o hotspot_tiling_emu

hotspot_mono_hw: $(SOURCES) $(RESOURCES)
	$(CC) $(ARGS) $(MONOTILE) $(HW) $(SOURCES) -o hotspot_mono_hw

hotspot_mono_hw.report.tar.gz: $(SOURCES) $(RESOURCES)
	rm -f hotspot_mono_hw
	$(CC) $(ARGS) $(MONOTILE) $(HW) -fsycl-link $(SOURCES) -o hotspot_mono_hw
	tar -caf hotspot_mono_hw.report.tar.gz hotspot_mono_hw.prj/reports

hotspot_tiling_hw: $(SOURCES) $(RESOURCES)
	$(CC) $(ARGS) $(TILING) $(HW) $(SOURCES) -o hotspot_tiling_hw

hotspot_tiling_hw.report.tar.gz: $(SOURCES) $(RESOURCES)
	rm -f hotspot_tiling_hw
	$(CC) $(ARGS) $(TILING) $(HW) -fsycl-link $(SOURCES) -o hotspot_tiling_hw
	tar -caf hotspot_tiling_hw.report.tar.gz hotspot_tiling_hw.prj/reports

hotspot_rodinia: Makefile
	./scripts/build_rodinia.sh

clean:
	git clean -dXf
