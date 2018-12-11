# Use the MPI wrapper for the gcc compiler:
CC = mpiCC

# Target Files
TARGET = nqueens

all: $(TARGET)

nqueens: nqueens.C
	$(CC) -g -Wall -o nqueens nqueens.C -lm

clean:
	$(RM) *.o *~ $(TARGET)
