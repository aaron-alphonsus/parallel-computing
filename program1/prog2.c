#include <stdlib.h>
#include <stdio.h>

void Usage(char* prog_name);

int main(int argc, char* argv[])
{
    long long n;

    if (argc != 2) 
        Usage(argv[0]);
    n = strtoll(argv[1], NULL, 10);
    if (n < 2) 
        Usage(argv[0]);
        
    return 0;
}

/*------------------------------------------------------------------
 * Function:  Usage
 * Purpose:   Print a message explaining how to run the program
 * In arg:    prog_name
 */
void Usage(char* prog_name) 
{
   fprintf(stderr, "Usage: %s <n>\n", prog_name);  /* Change */
   fprintf(stderr, "    Finds all primes less than or equal to n >= 2\n");
   exit(0);
}    
