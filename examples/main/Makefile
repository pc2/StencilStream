CC = dpcpp
STENCIL_PATH = ./

ARGS = -fintelfpga -Xsv -Xsfpc -std=c++17 -DSTENCIL_INDEX_WIDTH=32 -I$(STENCIL_PATH) 

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

ifdef AOCL_BOARD_PACKAGE_ROOT
	ARGS += -Xsboard=$(FPGA_BOARD_NAME) -Xsboard-package=$(AOCL_BOARD_PACKAGE_ROOT)
endif

SYNTH_ARGS = $(ARGS)
ifdef HARDWARE
	SYNTH_ARGS += -Xshardware
endif

REPORT_ARGS = $(ARGS) -fsycl-link -Xshardware

main: main.cpp Makefile
	$(CC) $(SYNTH_ARGS) main.cpp -o main

main.report.tar.gz: main.cpp Makefile
	$(CC) $(REPORT_ARGS) main.cpp -o main
	tar -caf main.report.tar.gz main.prj/reports

clean:
	git clean -dXf
