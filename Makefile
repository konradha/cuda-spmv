CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

CXXFLAGS := -std=c++17 -Xcompiler -fPIC
LDFLAGS := -L$(CUDA_HOME)/lib64 -lcusparse -lcudart -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64
INCLUDES := -Iinclude

KERNELS := baseline
OUTDIR := build
BINDIR := bin

KERNEL_OBJS := $(addprefix $(OUTDIR)/kernel_, $(addsuffix .o,$(KERNELS)))
DRIVERS := $(addprefix $(BINDIR)/spmv_, $(KERNELS))

SRCS_COMMON := src/mmio.cpp src/cusparse_helpers.cpp
OBJ_COMMON := $(patsubst src/%.cpp,$(OUTDIR)/%.o,$(SRCS_COMMON))

.PHONY: all clean
all: $(DRIVERS)

$(BINDIR)/spmv_%: $(OUTDIR)/driver.o $(OUTDIR)/kernel_%.o $(OBJ_COMMON)
	@mkdir -p $(BINDIR)
	$(NVCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OUTDIR)/driver.o: src/driver.cpp include/spmv.h
	@mkdir -p $(OUTDIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OUTDIR)/kernel_%.o: kernels/%.cu include/spmv.h
	@mkdir -p $(OUTDIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OUTDIR)/%.o: src/%.cpp include/spmv.h
	@mkdir -p $(OUTDIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OUTDIR) $(BINDIR)
