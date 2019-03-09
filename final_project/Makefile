NVCC        = nvcc #~-std=c++11
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 
EXE	        = exe
OBJ	        = main.o multikernel.o
OBJ			= main.cu
ARCH		= -arch=sm_52
CJSON		= -l cjson

default: $(EXE)

# support.o: support.cu support.h
# 	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

multikernel.o: multikernel.cu
	$(NVCC) -c -o $@ multikernel.cu $(NVCC_FLAGS)

main.o: main.cu multikernel.o
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)


$(EXE): $(OBJ)
	$(NVCC) $(CJSON) $(ARCH) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)