/* File:  
 *    prog2.c
 *
 * Purpose:
 *    A parallel implementation of the sieve of Eratosthenes algorithm using 
 *    OpenMP timing functions to compare speed-up achieved.
 *
 * Input:
 *    unsigned long long n (>=2)
 * Output:
 *    Prime numbers less than or equal to n
 *
 * Compile: 
 *    gcc -g -Wall -fopenmp -o prog2 prog2.c -lm 
 *    OR make prog2
 * Usage:   
 *    ./prog2 <n> <print> <reps>
 *
 * Professor: 
 *    Dr. Christer Karlsson
 * Authors:   
 *    Aaron Alphonsus 
 * Class:     
 *    CSC410 - Parallel Computing 
 */
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

/*------------------------------------------------------------------------------
 * Function: usage
 *
 * Purpose:  Print a message explaining how to run the program
 *
 * In args:  prog_name
 *
 * Returns:  void
 */
void usage(char* prog_name) 
{
    fprintf(stderr, "Usage: %s <n> <print> <reps>\n", prog_name); 
    fprintf(stderr, "    Finds all primes less than or equal to n (n >= 2).\n");
    fprintf(stderr, "    Enter 1 or 0 to print or suppress printing.\n");
    fprintf(stderr, "    Number of repititions: reps >= 1"
        " (for timing purposes).\n");

    exit(0);
}

/*------------------------------------------------------------------------------
 * Function: time_parallel
 *
 * Purpose:  Parallelizes step 3 of the sieve of Eratosthenes algorithm
 *           Keeps track of how long this portion of the algorithm takes
 *           Runs the algorithm reps times to get an average running time
 *
 * In args:  reps, n, primes
 *
 * Out args: primes, time
 *
 * Returns:  time/reps: Time to run the algorithm averaged over reps times
 */
double time_parallel(int reps, unsigned long long n, unsigned long long* primes)
{
    int i, j, k, thread_count = 8;
    double begin, end, time = 0.0; 

    // runs the algorithm reps times so that we can get an average run time
    for(k = 0; k < reps; k++) 
    {  
        for(i = 0; i+2 <= sqrt(n); i++)
        {
            if(primes[i] == 1)
            {
                begin = omp_get_wtime();
               
                // Finds all multiples of i+2 in parallel 
                # pragma omp parallel for num_threads(thread_count) \
                    default(none) private(j) shared(i, n, primes) \
                    // schedule(static, 100)
                for(j = 2; j <= n/(i+2); j++) 
                    primes[(i+2)*j - 2] = 0;
                
                end = omp_get_wtime();
                time += (double)((end-begin)*1000.0); 
            }
        }
    }
    return time/reps;
} 
 
/*------------------------------------------------------------------------------
 * Function: time_serial
 *
 * Purpose:  Runs sieve of Eratosthenes algorithm seially
 *           Keeps track of how long step 3 of the algorithm takes
 *           Runs the algorithm reps times to get an average running time
 *
 * In args:  reps, n, primes
 *
 * Out args: primes, time
 *
 * Returns:  time/reps: Time to run the algorithm averaged over reps times
 */
double time_serial(int reps, unsigned long long n, unsigned long long* primes)
{
    int i, j, k;
    double begin, end, time = 0.0;
    
    // runs the algorithm reps times so that we can get an average run time
    for(k = 0; k < reps; k++) 
    {  
        for(i = 0; i+2 <= sqrt(n); i++)
        {
            if(primes[i] == 1)
            {
                begin = omp_get_wtime(); 
                
                // Finds all multiples of i+2 serially
                for(j = 2; j <= n/(i+2); j++) 
                    primes[(i+2)*j - 2] = 0;

                end = omp_get_wtime();
                time += (double)((end-begin)*1000.0); 
            }
        }
    }
    return time/reps; 
}

/*------------------------------------------------------------------------------
 * Function: output
 *
 * Purpose:  Calculates padding value to print with correct formatting
 *           Prints primes from 2 to n with 10 primes on each row
 *
 * In args:  n, primes
 *
 * Returns:  void
 */
void output(unsigned long long n, unsigned long long* primes)
{
    unsigned long long i, num_primes = 0, every_ten = 0;     
    int padding; 

    // find number of primes to calculate padding amount
    for(i = 0; i <= n-2; i++)
        if(primes[i] == 1)
            num_primes++;
    // calculate padding amount to output first column with right spacing 
    if(num_primes == 1)
        padding = 1;
    else
        padding = floor(log10(num_primes-1)) + 1;

    // print all primes >=2 and <=n 
    printf("%*d: ", padding, 0);
    for(i = 0; i <= n-2; i++)
    {
        if(primes[i] == 1)
        {
            // print indicator of every group of 10 with the appropriate padding
            if(every_ten%10 == 0 && every_ten > 0)
                printf("\n%*lld: ", padding, every_ten); 
            printf("%lld ", i+2); // primes separated by spaces
            every_ten++;
        }
    }
    printf("\n");
}

/*------------------------------------------------------------------------------
 * Function: output_testing
 *
 * Purpose:  Printing formatted to match output of test primes file downloaded
 *
 * In args:  n, primes
 *
 * Returns:  void
 */
void output_testing(unsigned long long n, unsigned long long* primes)
{
    unsigned long long i; 
    int chars76 = 76;

    for(i = 0; i <= n-2; i++)
    {
        if(primes[i] == 1)
        {
            // the test file prints a maximum of 76 characters on each line
            // makes sure a newline gets inserted on reaching the 76th character 
            chars76 -= (floor(log10(i+2))+2);
            if(chars76 < 0)
            {
                printf("\n");
                chars76 = 76 - (floor(log10(i+2))+2);
            } 
            printf("%lld,", i+2); // primes separated by commas
        }
    }
    printf("\n");
}

/*------------------------------------------------------------------------------
 * Function: main
 *
 * Purpose:  Checks command-line arguments and calls usage function if invalid
 *           Declares variables, initializes list of primes with 1s
 *           Calls parallel and serial functions to run and time each method 
 *           Calls printing function and prints time taken by each method
 *
 * In args:  argc, argv
 *
 * Out args: time_par, time_ser
 *
 * Returns:  0 indicating normal termination
 */
int main(int argc, char* argv[])
{
    unsigned long long n;
    int print = 0, reps = 10;

    // check if command line arguments are valid
    if(argc != 4) 
        usage(argv[0]);
    
    n = strtoull(argv[1], NULL, 10);
    print = strtol(argv[2], NULL, 10);
    reps = strtol(argv[3], NULL, 10);
 
    if(n < 2) 
        usage(argv[0]);
    if(print != 1 && print != 0)
        usage(argv[0]);
    if(reps < 1)
        usage(argv[0]);
 
    // declare variables and call functions to time each algorithm
    unsigned long long i;
    unsigned long long *primes = malloc(n * sizeof *primes); 
    double time_par, time_ser;

    for(i = 0; i <= n-2; i++) // initialize prime list with 1s
        primes[i] = 1;  

    time_par = time_parallel(reps, n, primes); 
    time_ser = time_serial(reps, n, primes); 

    // comment out output_testing if other formatting required 
    if(print) 
    {
        output(n, primes);
        // output_testing(n, primes);
    }     

    free(primes);

    printf("Parallel time = %lf ms\n", time_par);
    printf("Serial time = %lf ms\n", time_ser);
    
    return 0;
}
