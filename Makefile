CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc

CXXFLAGS := -std=c++17 -Xcompiler -fPIC
LDFLAGS := -L$(CUDA_HOME)/lib64 -lcusparse -lcudart -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64
INCLUDES := -Iinclude

OUTDIR := build
BINDIR := bin

# Auto-discover kernels and compile them all into one binary
KERNEL_SRCS  := $(wildcard kernels/*.cu)
KERNEL_NAMES := $(basename $(notdir $(KERNEL_SRCS)))
KERNEL_OBJS  := $(addprefix $(OUTDIR)/kernel_, $(addsuffix .o,$(KERNEL_NAMES)))

APP := $(BINDIR)/spmv_all

SRCS_COMMON := src/mmio.cpp src/cusparse_helpers.cpp
OBJ_COMMON  := $(patsubst src/%.cpp,$(OUTDIR)/%.o,$(SRCS_COMMON))

.PHONY: all clean list
all: $(APP)

$(APP): $(OUTDIR)/driver.o $(OBJ_COMMON) $(KERNEL_OBJS)
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

list:
	@echo "Kernels:" $(KERNEL_NAMES)

clean:
	rm -rf $(OUTDIR) $(BINDIR)

