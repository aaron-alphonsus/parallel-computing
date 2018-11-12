# Use the NVIDIA C compiler:
CC = nvcc

# Target Files
TARGET = prog2

all: $(TARGET)

prog2: prog2.cu
	$(CC) -o prog2 prog2.cu

clean:
	$(RM) *.o *~ $(TARGET)
