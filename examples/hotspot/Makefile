# Copyright © 2020-2021Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
CC = dpcpp
STENCIL_STREAM_PATH = ./

ARGS = -std=c++20 -I$(STENCIL_STREAM_PATH) -O3 -qactypes

ifdef EBROOTGCC
	ARGS += --gcc-toolchain=$(EBROOTGCC)
endif

HW = $(ARGS) -fintelfpga -Xshardware -Xsv
MONOTILE = $(HW) -DEXECUTOR=0
TILING = $(HW) -DEXECUTOR=1

ifdef AOCL_BOARD_PACKAGE_ROOT
	HW += -Xsboard=$(FPGA_BOARD_NAME) -Xsboard-package=$(AOCL_BOARD_PACKAGE_ROOT)
endif

SOURCES = hotspot.cpp
RESOURCES = $(wildcard StencilStream/*) Makefile

all: hotspot_cpu hotspot_rodinia hotspot_mono hotspot_tiling

hotspot_cpu: $(SOURCES) $(RESOURCES)
	$(CC) $(ARGS) $(SOURCES) -DEXECUTOR=2 -o hotspot_cpu

hotspot_mono: $(SOURCES) $(RESOURCES)
	$(CC) $(MONOTILE) $(SOURCES) -reuse-exe=hotspot_mono -o hotspot_mono

hotspot_mono.report.tar.gz: $(SOURCES) $(RESOURCES)
	rm -f hotspot_mono
	$(CC) $(MONOTILE) -fsycl-link $(SOURCES) -o hotspot_mono
	tar -caf hotspot_mono.report.tar.gz hotspot_mono.prj/reports

hotspot_tiling: $(SOURCES) $(RESOURCES)
	$(CC) $(TILING) $(SOURCES) -reuse-exe=hotspot_tiling -o hotspot_tiling

hotspot_tiling.report.tar.gz: $(SOURCES) $(RESOURCES)
	rm -f hotspot_tiling
	$(CC) $(TILING) -fsycl-link $(SOURCES) -o hotspot_tiling
	tar -caf hotspot_tiling.report.tar.gz hotspot_tiling.prj/reports

hotspot_rodinia: Makefile
	./scripts/build_rodinia.sh

clean:
	git clean -dXf
