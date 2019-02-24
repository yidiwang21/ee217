NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = kernel_sched
# OBJ	        = main.o launcher.o support.o
OBJ			= main.cu
ARCH		= -arch=sm_61
CJSON		= -l cjson

default: $(EXE)

# support.o: support.cu support.h
# 	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

# launcher.o: launcher.cu 
# 	$(NVCC) -c -o $@ launcher.cu $(NVCC_FLAGS)

# main.o: main.cu 
# 	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)


$(EXE): $(OBJ)
	$(NVCC) $(CJSON) $(ARCH) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)