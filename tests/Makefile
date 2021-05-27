# Copyright © 2020-2021 Jan-Oliver Opdenhövel, Paderborn Center for Parallel Computing, Paderborn University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
CC = dpcpp
STENCIL_PATH = ../

ARGS = -std=c++20 -g -I$(STENCIL_PATH) -fintelfpga -DSTENCIL_INDEX_WIDTH=32

ifdef OPTIMIZE
	ARGS += -O3
endif

ifdef EBROOTGCC
	ARGS += --gcc-toolchain=$(EBROOTGCC)
endif

UNIT_ARGS = $(ARGS)
UNIT_OBJECTS = $(patsubst %.cpp,%.o,$(wildcard src/units/*.cpp))
RESOURCES = src/res/*.hpp ../StencilStream/* Makefile

SYNTH_ARGS = $(ARGS) -Xsv

unit_test: $(UNIT_OBJECTS) $(RESOURCES)
	$(CC) $(UNIT_ARGS) -Xsv $(UNIT_OBJECTS) -o unit_test

src/units/%.o: src/units/%.cpp $(RESOURCES)
	$(CC) $(UNIT_ARGS) -c $< -o $@

src/units/main.o: src/units/main.cpp src/res/*.hpp Makefile
	$(CC) $(UNIT_ARGS) -c $< -o $@

synthesis_emu: src/synthesis/main.cpp $(RESOURCES)
	$(CC) $(SYNTH_ARGS) src/synthesis/main.cpp -o synthesis_emu

synthesis_hw: src/synthesis/main.cpp $(RESOURCES)
	$(CC) $(SYNTH_ARGS) -DHARDWARE -Xshardware src/synthesis/main.cpp -o synthesis_hw

synthesis_hw.report.tar.gz: src/synthesis/main.cpp $(RESOURCES)
	rm -f synthesis_hw
	$(CC) $(SYNTH_ARGS) -DHARDWARE -fsycl-link -Xshardware src/synthesis/main.cpp -o synthesis_hw
	tar -caf synthesis_hw.report.tar.gz synthesis_hw.prj/reports

clean:
	git clean -dXf