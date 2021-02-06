# Copyright © 2020 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
CC = dpcpp
STENCIL_PATH = ./

ARGS = -std=c++17 -DSTENCIL_INDEX_WIDTH=32 -I$(STENCIL_PATH) 

ifdef EBROOTGCC
	ARGS += --gcc-toolchain=$(EBROOTGCC)
endif

ifndef PIPELINE_LEN
	PIPELINE_LEN=10
endif
ARGS += -DSTENCIL_PIPELINE_LEN=$(PIPELINE_LEN)

ifdef HARDWARE
	ARGS += -DHARDWARE
endif

SYNTH_ARGS = $(ARGS) -fintelfpga -Xsv
ifdef HARDWARE
	SYNTH_ARGS += -Xshardware
endif

ifdef AOCL_BOARD_PACKAGE_ROOT
	SYNTH_ARGS += -Xsboard=$(FPGA_BOARD_NAME) -Xsboard-package=$(AOCL_BOARD_PACKAGE_ROOT)
endif

REPORT_ARGS = $(SYNTH_ARGS) -fsycl-link -Xshardware

hotspot.ll: hotspot.cpp Makefile
	$(CC) $(ARGS) -fsycl-targets=spir64-unknown-linux-sycldevice -c hotspot.cpp -o hotspot.o
	$(CC) $(ARGS) -fsycl-link-targets=spir64-unknown-linux-sycldevice hotspot.o -o hotspot.spv
	llvm-spirv -r hotspot.spv
	llvm-dis hotspot.bc

hotspot: hotspot.cpp Makefile
	$(CC) $(SYNTH_ARGS) hotspot.cpp -o hotspot

hotspot.report.tar.gz: hotspot.cpp Makefile
	$(CC) $(REPORT_ARGS) hotspot.cpp -o hotspot.a
	tar -caf hotspot.report.tar.gz hotspot.prj/reports

clean:
	git clean -dXf
