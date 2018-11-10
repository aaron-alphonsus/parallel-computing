# Use the GNU C compiler:                                                    
CC = gcc                                                                      

#TARGET FILES 
TARGET = prog1 prog2
                                                                                
all: $(TARGET)                                                                  
                                                                                 
prog1: prog1.c                                                           
	$(CC) -g -Wall -fopenmp -o prog1 prog1.c                                              
                                                                                 
prog2: prog2.c                                                    
	$(CC) -g -Wall -fopenmp -o prog2 prog2.c -lm                                         
                                                                                 
clean:                                                                          
	$(RM) *.o *~  $(TARGET)
