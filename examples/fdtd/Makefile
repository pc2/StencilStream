CC = dpcpp
STENCIL_PATH = ./

ARGS = -fintelfpga -Xsv -std=c++17 -DSTENCIL_INDEX_WIDTH=64 -I$(STENCIL_PATH) -lpthread -lz

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

fdtd: src/*.cpp src/*.hpp Makefile
	$(CC) $(SYNTH_ARGS) src/*.cpp -o fdtd

fdtd.report.tar.gz: src/*.cpp src/*.hpp Makefile
	$(CC) $(REPORT_ARGS) src/*.cpp -o fdtd.a
	tar -caf fdtd.report.tar.gz fdtd.prj/reports

clean:
	git clean -dXf
