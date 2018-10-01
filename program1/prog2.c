#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

void Usage(char* prog_name);

int main(int argc, char* argv[])
{
    long long n;

    if (argc != 2) 
        Usage(argv[0]);
    n = strtoull(argv[1], NULL, 10);
    if (n < 2) 
        Usage(argv[0]);

    unsigned long long i, j, num_primes = 0, every_ten = 0;
    int thread_count = 4; 
    // printf("%d, %d\n", omp_get_num_procs(), omp_get_max_threads());

    int padding;
    unsigned long long primes[n-1];

    # pragma omp parallel for num_threads(thread_count) 
    for(i = 0; i <= n-2; i++)
        primes[i] = 1; 
   
    int k, reps = 100;
    double begin, end;
    double time_par = 0.0, time_ser = 0.0;

    for(k = 0; k < reps; k++) 
    {  
        begin = omp_get_wtime(); 
        for(i = 0; i+2 <= sqrt(n); i++)
        {
            // printf("(i, primes[i]) = (%lld, %lld)", i, primes[i]);
            if(primes[i] == 1)
            {
                # pragma omp parallel for num_threads(thread_count) \
                    default(none) private(j) shared(i, n, primes)
                for(j = 2; j <= n/(i+2); j++) 
                    primes[(i+2)*j - 2] = 0;
            }
        }
        end = omp_get_wtime();
        time_par += (double)((end-begin)*1000.0); 
    }

    for(k = 0; k < reps; k++) 
    {  
        begin = omp_get_wtime(); 
        for(i = 0; i+2 <= sqrt(n); i++)
        {
            // printf("(i, primes[i]) = (%lld, %lld)", i, primes[i]);
            if(primes[i] == 1)
            {
                for(j = 2; j <= n/(i+2); j++) 
                    primes[(i+2)*j - 2] = 0;
            }
        }
        end = omp_get_wtime();
        time_ser += (double)((end-begin)*1000.0); 
    }

     
    // For printing purposes
    for(i = 0; i <= n-2; i++)
        if(primes[i] == 1)
            num_primes++;
    // printf("%lld, %lld\n", num_primes, 
    //    (long long)floor(log10(num_primes-1))+1);
    padding = floor(log10(num_primes-1)) + 1;

    printf("%*d: ", padding, 0);
    for(i = 0; i <= n-2; i++)
    {
        // printf("i = %lld, ", i);
        if(primes[i] == 1)
        {
            if(every_ten%10 == 0 && every_ten > 0)
                printf("\n%*lld: ", padding, every_ten);
            printf("%lld ", i+2);
            every_ten++;
        }
    }

    // // Printing to match output of test files to run diff on
    // int chars76 = 76;
    // for(i = 0; i <= n-2; i++)
    // {
    //     if(primes[i] == 1)
    //     {
    //         chars76 -= (floor(log10(i+2))+2);
    //         if(chars76 < 0)
    //         {
    //             printf("\n");
    //             chars76 = 76 - (floor(log10(i+2))+2);
    //         } 
    //         printf("%lld,", i+2);
    //         every_ten++;
    //     }
    // }
 
    printf("\nParallel time = %lf ms\n", (double)time_par / reps);
    printf("Serial time = %lf ms\n", (double)time_ser / reps);
    
    return 0;
}

// Separate out print function

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
