#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/*------------------------------------------------------------------
 * Function:  usage
 * Purpose:   Print a message explaining how to run the program
 * In arg:    prog_name
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

double time_parallel(int reps, unsigned long long n, unsigned long long* primes)
{
    int i, j, k, thread_count = 8;
    double begin, end, time = 0.0; 

    for(k = 0; k < reps; k++) 
    {  
        for(i = 0; i+2 <= sqrt(n); i++)
        {
            if(primes[i] == 1)
            {
                begin = omp_get_wtime();
                
                # pragma omp parallel for num_threads(thread_count) \
                    default(none) private(j) shared(i, n, primes) \
                    // schedule(dynamic, 100)
                for(j = 2; j <= n/(i+2); j++) 
                    primes[(i+2)*j - 2] = 0;
                
                end = omp_get_wtime();
                time += (double)((end-begin)*1000.0); 
            }
        }
    }
    return time/reps;
} 
 
double time_serial(int reps, unsigned long long n, unsigned long long* primes)
{
    int i, j, k;
    double begin, end, time = 0.0;
    
    for(k = 0; k < reps; k++) 
    {  
        for(i = 0; i+2 <= sqrt(n); i++)
        {
            if(primes[i] == 1)
            {
                begin = omp_get_wtime(); 

                for(j = 2; j <= n/(i+2); j++) 
                    primes[(i+2)*j - 2] = 0;

                end = omp_get_wtime();
                time += (double)((end-begin)*1000.0); 
            }
        }
    }
    return time/reps; 
}

void output(unsigned long long n, unsigned long long* primes)
{
    unsigned long long i, num_primes = 0, every_ten = 0;     
    int padding; 

    for(i = 0; i <= n-2; i++)
        if(primes[i] == 1)
            num_primes++;
    
    if(num_primes == 1)
        padding = 1;
    else
        padding = floor(log10(num_primes-1)) + 1;

    printf("%*d: ", padding, 0);
    for(i = 0; i <= n-2; i++)
    {
        if(primes[i] == 1)
        {
            if(every_ten%10 == 0 && every_ten > 0)
                printf("\n%*lld: ", padding, every_ten);
            printf("%lld ", i+2);
            every_ten++;
        }
    }
    printf("\n");
}

void output_testing(unsigned long long n, unsigned long long* primes)
{
    // Printing to match output of test files to run diff on
    unsigned long long i; 
    int chars76 = 76;

    for(i = 0; i <= n-2; i++)
    {
        if(primes[i] == 1)
        {
            chars76 -= (floor(log10(i+2))+2);
            if(chars76 < 0)
            {
                printf("\n");
                chars76 = 76 - (floor(log10(i+2))+2);
            } 
            printf("%lld,", i+2);
        }
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    unsigned long long n;
    int print = 0, reps = 10;

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

    unsigned long long i;
    unsigned long long *primes = malloc(n * sizeof *primes); 
    double time_par, time_ser;

    for(i = 0; i <= n-2; i++)
        primes[i] = 1;  

    time_par = time_parallel(reps, n, primes); 
    time_ser = time_serial(reps, n, primes); 
 
    if(print) 
    {
        output(n, primes);
        // output_testing(n, primes);
    }     

    printf("Parallel time = %lf ms\n", time_par);
    printf("Serial time = %lf ms\n", time_ser);
    
    return 0;
}
