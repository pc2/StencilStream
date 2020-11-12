CC = dpcpp
STENCIL_PATH = ./

ARGS = -fintelfpga -Xsv -Xsfpc -std=c++17 -I$(STENCIL_PATH) 

ifdef EBROOTGCC
	ARGS += --gcc-toolchain=$(EBROOTGCC)
endif

ifndef PIPELINE_LEN
	PIPELINE_LEN=10
endif
ARGS += -DSTENCIL_PIPELINE_LEN=$(PIPELINE_LEN)

ifdef HARDWARE
	ARGS += -DHARDWARE
	ARGS += -Xshardware
endif

ifdef AOCL_BOARD_PACKAGE_ROOT
	ARGS += -Xsboard=$(FPGA_BOARD_NAME) -Xsboard-package=$(AOCL_BOARD_PACKAGE_ROOT)
endif

conway: clean conway.cpp Makefile
	$(CC) $(ARGS) conway.cpp -o conway

conway.report.tar.gz: clean conway.cpp Makefile
	$(CC) $(ARGS) -fsycl-link -Xshardware conway.cpp -o conway
	tar -caf conway.report.tar.gz conway.prj/reports

clean:
	git clean -dXf