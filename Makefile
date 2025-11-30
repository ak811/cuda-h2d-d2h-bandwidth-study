NVCC       := nvcc
NVCC_FLAGS := -O2

SRCS := main.cu bandwidth.cu
HDRS := bandwidth.h cuda_utils.cuh

TARGET := bandwidth

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET) bandwidth.csv bandwidth_plot.png
