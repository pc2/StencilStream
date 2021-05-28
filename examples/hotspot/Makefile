# Copyright © 2020-2021Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
CC = dpcpp
STENCIL_STREAM_PATH = ./

ARGS = -fintelfpga -Xsv -Xsprofile -std=c++17 -I$(STENCIL_STREAM_PATH) -O3

ifdef EBROOTGCC
	ARGS += --gcc-toolchain=$(EBROOTGCC)
endif

ifdef AOCL_BOARD_PACKAGE_ROOT
	ARGS += -Xsboard=$(FPGA_BOARD_NAME) -Xsboard-package=$(AOCL_BOARD_PACKAGE_ROOT)
endif

EMU_ARGS = $(ARGS)
HW_ARGS = $(ARGS) -DHARDWARE -Xshardware 

hotspot_emu: hotspot.cpp Makefile
	$(CC) $(EMU_ARGS) hotspot.cpp -o hotspot_emu

hotspot_hw: hotspot.cpp Makefile
	$(CC) $(HW_ARGS) hotspot.cpp -o hotspot_hw

hotspot_hw.report.tar.gz: hotspot.cpp Makefile
	rm -f hotspot_hw
	$(CC) $(HW_ARGS) -fsycl-link hotspot.cpp -o hotspot_hw
	tar -caf hotspot_hw.report.tar.gz hotspot_hw.prj/reports

hotspot_rodinia: Makefile
	./scripts/build_rodinia.sh

clean:
	git clean -dXf