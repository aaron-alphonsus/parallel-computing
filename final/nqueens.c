#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

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

int* ithPermutation(int n, unsigned long long i)
{
    int j, k = 0;
    unsigned long long quo;
    int *perm = (int *)calloc(n, sizeof(int));
    // unsigned long long *fact = (unsigned long long *)calloc(n, sizeof(unsigned long long));
    // int *factoradic = (int *)calloc(n, sizeof(int));
    // std::vector<int> sequence;
 
    k = 1;
    quo = i;
    while(quo / (k+1) > 0)
    {
        k++;
        // printf("%d %d %d\n", quo / k, k, quo % k); 
        perm[n-k] = quo % k;
        quo /= k;
    }
    k++;
    perm[n-k] = quo % k;
    // printf("%d %d %d\n", quo / k, k, quo % k); 

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

    // for(j = 0; j < n; j++)
    //     sequence.push_back(j);    
    // for(j = 0; j < n; j++)
    // {
    //     perm[j] = sequence[factoradic[j]];
    //     sequence.erase(sequence.begin() + factoradic[j]);
    // }         

    // // print permutation
    // for (k = 0; k < n; ++k)
    //     printf("%d ", perm[k]);
    // printf("\n");

    // free(factoradic);

    return perm;
}

int checkDiag(int n, int perm[])
{
    int i = 0, j = 0;
    int sameDiag = 0;
    int verticalDist, horizontalDist;

    while(i < n && sameDiag == 0)
    {
        j = i + 1;
        while(j < n && sameDiag == 0)
        {
            // printf("(perm[i], i), (perm[j], j) = " 
            //          "(%d, %d), (%d, %d)\n", perm[i], i, perm[j], j);
            verticalDist = abs(perm[i] - perm[j]);
            horizontalDist = abs(i - j);
            // printf("ver, hor = %d, %d\n", verticalDist, horizontalDist); 
            
            if(verticalDist == horizontalDist)
                sameDiag++; 

            j++;
        }
        i++; 
    } 

    return !sameDiag;
}

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

    int id, p, is_soln = 0, min_chunk = 2;
    unsigned long long nfact = 1; // Good enough till n = 20
    unsigned long long chunk; 
    
    int *perm = (int *)calloc(n, sizeof(int));
 
    unsigned long long subtotal = 0, grand_total = 0;
    double elapsed_time; /* Time to find, count solutions */

    for(int i = 1; i < n+1; i++)
        nfact *= i;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    chunk = fmax(min_chunk, floor((float) nfact / p));   
    // if(id == 0)
    //     printf("chunk = %lld", chunk);
 
    /* Start timer */
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    // printf("Process %d range = [%lld, %lld]\n", id, chunk*id, chunk*(id+1));

    for(unsigned long long i = chunk*id; i < nfact && i < chunk*(id+1); i++)
    {    
        if(i != chunk*id)
            std::next_permutation(perm, perm + n);        
        else 
            perm = ithPermutation(n, i);
        
        is_soln += checkDiag(n, perm);
        if(is_soln)
        {
            subtotal += is_soln;
            is_soln = 0;
       
            // print permutation
            for (int k = 0; k < n; ++k)
                printf("%d ", perm[k]);
            printf("\n");
        }

        // printf("Process %d: %lld\n", id, i); 
    } 
    if(id == p-1)
    {
        for(unsigned long long i = chunk*p; i < nfact; i++)   
        {
            std::next_permutation(perm, perm + n); 
            is_soln += checkDiag(n, perm);    
            if(is_soln)
            {
                subtotal += is_soln;
                is_soln = 0;
       
                // print permutation
                for (int k = 0; k < n; ++k)
                    printf("%d ", perm[k]);
                printf("\n");
            }

            // printf("Process %d: %lld\n", id, i); 
        }
    } 

    MPI_Reduce(&subtotal, &grand_total, 1, MPI_UNSIGNED_LONG_LONG, 
        MPI_SUM , 0, MPI_COMM_WORLD);
    
    MPI_Barrier (MPI_COMM_WORLD);
    /* Stop timer */
    elapsed_time += MPI_Wtime(); /* elapsed time=current time-start time */

    if (0 == id) {
        // printf("p = %d\n", p);
        printf ("%lld\n", grand_total ); 
        printf ("Execution time %8.3f ms\n", 1000*elapsed_time);
        fflush (stdout);
    }

    MPI_Finalize();
    
    free(perm);
    
    return 0;
}
