CC=nvcc
ARCH=-arch=sm_52
SOURCES=main.cu
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=kernel_sched
CJSON=-l cjson
all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(CJSON) $(ARCH) $(OBJECTS) -o $@
.PHONY : clean
clean :
	-rm $(EXECUTABLE)