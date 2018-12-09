#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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
    fprintf(stderr, "Usage: %s <n> <print> \n", prog_name); 
    fprintf(stderr, "    Number of queens, size of chessboard (n*n).\n");
    fprintf(stderr, "    Enter 1 or 0 to print or suppress printing of" 
        " solutions.\n");
    
    exit(0);
}

int ithPermutation(const int n, int i)
{
    // Source: https://stackoverflow.com/questions/7918806/finding-n-th-permutation-without-computing-others/7919887#7919887

    int j, k = 0;
    int *fact = (int *)calloc(n, sizeof(int));
    int *perm = (int *)calloc(n, sizeof(int));
    int verticalDist, horizontalDist;
    int sameDiag = 0;
 
    // TODO: Change this method to divide with integers 1, 2, ... until 
    //  0 remainder 

    // compute factorial numbers
    fact[k] = 1;
    while (++k < n)
        fact[k] = fact[k - 1] * k;

    // compute factorial code
    for (k = 0; k < n; ++k)
    {
        perm[k] = i / fact[n - 1 - k];
        i = i % fact[n - 1 - k];
    }

    // // factoradic representation
    // for (k = 0; k < n; ++k)
    //     printf("%d ", perm[k]);
    // printf("\n");

    // readjust values to obtain the permutation
    // start from the end and check if preceding values are lower
    for (k = n - 1; k > 0; --k)
        for (j = k - 1; j >= 0; --j)
            if (perm[j] <= perm[k])
                perm[k]++;

    // // print permutation
    // for (k = 0; k < n; ++k)
    //     printf("%d ", perm[k]);
    // printf("\n");

    j = 0;
    k = 0; 
    while(j < n && sameDiag == 0)
    {
        k = j + 1;
        while(k < n && sameDiag == 0)
        {
            // printf("(perm[j], j), (perm[k], k) = " 
            //          "(%d, %d), (%d, %d)\n", perm[j], j, perm[k], k);
            verticalDist = abs(perm[j] - perm[k]);
            horizontalDist = abs(j - k);
            // printf("ver, hor = %d, %d\n", verticalDist, horizontalDist); 
            
            if(verticalDist == horizontalDist)
                sameDiag++; 

            k++;
        }
        j++; 
    }      
    
    free(fact);
    free(perm);

    // printf("!sameDiag = %d\n", !sameDiag); 

    return !sameDiag;
}

// // Same macro as appeared in circuitsat1.c
// #define EXTRACT_BIT(n,i) ( ((n)&(1 <<(i)) ) ?1:0)
// 
// // check_circuit (id ,n) is same as in circuitsat1.c
// int check_circuit(int proc_id, int inputval);

int main (int argc, char* argv[]) 
{
    int n;
    int print = 0;

    // check if command line arguments are valid
    if(argc != 3) 
        usage(argv[0]);
    
    n = strtol(argv[1], NULL, 10);
    print = strtol(argv[2], NULL, 10);
 
    if(n < 1) 
        usage(argv[0]);
    if(print != 1 && print != 0)
        usage(argv[0]);

    int id, p;
    unsigned long long nfact = 1; // Good enough till n = 20
    // int chunk, min_chunk = 2; 
    
    int subtotal, grand_total;
    double elapsed_time; /* Time to find, count solutions */

    // Why don't I have to put an if statement with id = 0 around this?
    for(int i = 1; i < n+1; i++)
        nfact *= i; 

    MPI_Init (&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // chunk = fmax(min_chunk, ceil((float) nfact / p));   
 
    /* Start timer */
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    // for(unsigned long long i = chunk*id; i < nfact && i < chunk*(id+1); i++)
    //     printf("Process %d: %lld\n", id, i); 
    
    subtotal = 0;
    for(unsigned long long i = id; i < nfact; i += p)
    {
        // printf("Process %d: %lld\n", id, i); 
        subtotal += ithPermutation(n, i); 
    }  

    MPI_Reduce(&subtotal, &grand_total, 1, MPI_INT, 
        MPI_SUM , 0, MPI_COMM_WORLD);
    
    if(id == 0)
        printf ("%d\n", grand_total );    
 
    MPI_Barrier (MPI_COMM_WORLD);
    /* Stop timer */
    elapsed_time += MPI_Wtime(); /* elapsed time=current time-start time */

    if (0 == id) {
        printf ("Execution time %8.3f ms\n", 1000*elapsed_time);
        fflush (stdout);
    }

    MPI_Finalize();
    return 0;
}
