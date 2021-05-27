# Copyright © 2020-2021Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
CC = dpcpp
STENCIL_STREAM_PATH = ./

ARGS = -fintelfpga -Xsv -std=c++17 -DSTENCIL_INDEX_WIDTH=64 -DFDTD_BURST_SIZE=1024 -I$(STENCIL_STREAM_PATH) -O3 -lpthread

ifdef EBROOTGCC
	ARGS += --gcc-toolchain=$(EBROOTGCC)
endif

ifdef AOCL_BOARD_PACKAGE_ROOT
	ARGS += -Xsboard=$(FPGA_BOARD_NAME) -Xsboard-package=$(AOCL_BOARD_PACKAGE_ROOT)
endif

EMU_ARGS = $(ARGS)
HW_ARGS = $(ARGS) -DHARDWARE -Xshardware 

SOURCES = $(wildcard src/*.cpp)
RESOURCES = $(wildcard src/*.hpp) $(wildcard StencilStream/*) Makefile

fdtd_emu: $(SOURCES) $(RESOURCES)
	$(CC) $(EMU_ARGS) $(SOURCES) -o fdtd_emu

fdtd_hw: $(SOURCES) $(RESOURCES)
	$(CC) $(HW_ARGS) $(SOURCES) -o fdtd_hw

fdtd_hw.report.tar.gz: src/*.cpp src/*.hpp Makefile
	rm -f fdtd_hw
	$(CC) $(HW_ARGS) -fsycl-link $(SOURCES) -o fdtd_hw
	tar -caf fdtd_hw.report.tar.gz fdtd_hw.prj/reports

clean:
	git clean -dXf
