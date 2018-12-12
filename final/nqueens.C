/*------------------------------------------------------------------------------
 * File:
 *     nqueens.C
 *                                                                               
 * Purpose:
 *     Program to find all solutions to the n-queens problem. The user 
 *     inputs the size of the chessboard and indicates whether they want 
 *     solutions printed to the screen. The program generates and evaluates all 
 *     possible positions where queens do not share a row or column. It does 
 *     this in parallel using the MPI model. The program divides up chunks of 
 *     possible positions that a process can check allowing each to take 
 *     advantage of the next_permutation() function.  
 *                                                                               
 * Input:                                                                        
 *     unsigned long long n (>=1), int print (1 or 0)
 * Output:                                                                       
 *     Number of solutions for the n-queens problem, and if desired, the 
 *     positioning of the queens for each solution 
 *                                                                               
 * Compile:
 *     mpiCC -g -Wall -o nqueens nqueens.C -lm 
 *      OR 
 *     make
 * Usage:                                                                        
 *     ./nqueens <n> <print>
 *                                                                               
 * Professor:                                                                    
 *     Dr. Christer Karlsson                                                      
 * Authors:                                                                      
 *     Aaron Alphonsus                                                            
 * Class:                                                                        
 *     CSC410 - Parallel Computing                                                
 */

#include <algorithm>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

/*------------------------------------------------------------------------------
 * Function: usage
 *
 *  Purpose: Print a message explaining how to run the program.
 *
 *  In args: char* prog_name
 *
 *  Returns: void
 */
void usage(char* prog_name) 
{
    fprintf(stderr, "Usage: %s <n> <print> \n", prog_name); 
    fprintf(stderr, "    Number of queens, size of chessboard (n*n).\n");
    fprintf(stderr, "    Enter 1 or 0 to print or suppress printing of" 
        " solutions.\n");
    
    exit(0);
}

/*------------------------------------------------------------------------------
 * Function: 
 *     ithPermutation
 *
 * Purpose: 
 *     Given numbers n and i, this function calculates the ith permutation of 
 *     the array [0, 1, ..., n].  
 *     Based on - https://bit.ly/2tujfr1 and https://stackoverflow.com/a/7919887 
 *     Calculates the factoradic representation of i and uses it find the ith 
 *     permutation. 
 *
 *  In args: int n, unsigned long long i
 *
 * Out args: int* perm 
 *
 *  Returns: int* perm - the ith permutation of the array from 0 to n
 */
int* ithPermutation(int n, unsigned long long i)
{
    int j, k = 0;
    unsigned long long quo;

    // Allocate space for permutation array and initialize to 0
    int *perm = (int *)calloc(n, sizeof(int));
    
    // int *factoradic = (int *)calloc(n, sizeof(int));
    // std::vector<int> sequence;

    // Divide 'i' repeatedly by natural numbers until the quotient is 0
    // Yields the 'factoradic' representation of the decimal number i 
    k = 1;
    quo = i;
    while(quo / (k+1) > 0)
    {
        k++;
        perm[n-k] = quo % k; // Store result from back to front
        quo /= k;
    }
    k++;
    perm[n-k] = quo % k;

    // re-adjust values to obtain the permutation
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

    // free(factoradic);

    return perm;
}

/*------------------------------------------------------------------------------
 * Function: 
 *     checkDiag
 *
 * Purpose: 
 *     Checks every pair of queens of a potential solution to evaluate whether 
 *     they are on the same diagonal. Based on - 
 *     https://stackoverflow.com/a/3209201
 *
 *  In args: int n, int perm[]
 *
 * Out args: int sameDiag 
 *
 *  Returns: int sameDiag - 0 if no queens share a diagonal, 1 otherwise
 */
int checkDiag(int n, int perm[])
{
    int i = 0, j = 0;
    int sameDiag = 0;
    int verticalDist, horizontalDist;

    // For each pair of queens, check if horizontal distance is the same as 
    // vertical distance from each other. If so, the queens are on the same 
    // diagonal.
    while(i < n && sameDiag == 0)
    {
        j = i + 1;
        while(j < n && sameDiag == 0)
        {
            verticalDist = abs(perm[i] - perm[j]);
            horizontalDist = abs(i - j);
            
            if(verticalDist == horizontalDist)
                sameDiag++; 

            j++;
        }
        i++; 
    } 

    return !sameDiag;
}

/*------------------------------------------------------------------------------
 * Function: main
 *
 * Purpose:  
 *     Checks command-line arguments and calls usage function if invalid. 
 *     Defines how the problem is divided among the available processors. Each
 *     processor keeps a local count of the number of solutions found which is 
 *     later combined into a global total count using a reduction. Number of 
 *     solutions and execution time is printed to the console. Each valid 
 *     solution is also printed if the user sets the print command-line argument
 *
 * In args:  int argc, char* argv
 *
 * Out args: grand_total, perm
 *
 * Returns:  0 indicating normal termination
 */
int main (int argc, char* argv[]) 
{
    int n, print;

    // check if command line arguments are valid
    if(argc != 3) 
        usage(argv[0]);
    
    n = strtol(argv[1], NULL, 10);
    print = strtol(argv[2], NULL, 10);

    // Call usage function if invalid 
    if(n < 1) 
        usage(argv[0]);
    if(print != 1 && print != 0)
        usage(argv[0]);

    int id, p, is_soln = 0, min_chunk = 2;
    unsigned long long nfact = 1; // Good enough till n = 20
    unsigned long long chunk; 
   
    // Allocate space to hold permutation and initialize it with 0s 
    int *perm = (int *)calloc(n, sizeof(int));
 
    unsigned long long subtotal = 0, grand_total = 0;
    double elapsed_time; /* Time to find, count solutions */

    // Calculate n factorial - possible positions of queens not in the same row 
    // and column
    for(int i = 1; i < n+1; i++)
        nfact *= i;

    // Initialize MPI library, calculate ranks and total number of processes.
    MPI_Init (&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Divide the possible positions into chunks to assign to the processes
    chunk = fmax(min_chunk, floor((float) nfact / p));   
 
    /* Start timer */
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    // Each processor checks a chunk of permutations for valid solutions
    for(unsigned long long i = chunk*id; i < nfact && i < chunk*(id+1); i++)
    {
        // The first time, the process looks for the ith permutation, but every 
        // other time, it uses the quicker next_permutation function.    
        if(i != chunk*id)
            std::next_permutation(perm, perm + n);        
        else 
            perm = ithPermutation(n, i);
       
        // Once we have a potential solution, we check every pair of queens. 
        is_soln = checkDiag(n, perm);
        if(is_soln)
        {
            // If we've found a solution, we add it to the total number of 
            // solutions found by the process and do a reduction later to get
            // the total found by all the processes.
            subtotal += is_soln;
            is_soln = 0;
     
            // If the user passes in a 1 for their print flag, we print out 
            // every solution as we find it. 
            if(print) 
            {
                for (int k = 0; k < n; ++k)
                    printf("%d ", perm[k]);
                printf("\n");
            }
        }
    } 
    // After we divide the permutations and assign them to processors, there end
    // up being a remainder (because we can't assign fractional tasks, or rather
    // we don't want to assign fractional tasks) 
    // There is definitely a better mapping than just assigning them all to the
    // last processor so this is an area we need to make more efficient TODO
    if(id == p-1)
    {
        for(unsigned long long i = chunk*p; i < nfact; i++)   
        {
            std::next_permutation(perm, perm + n); 
            is_soln = checkDiag(n, perm);    
            if(is_soln)
            {
                subtotal += is_soln;
                is_soln = 0;
       
                if(print) 
                {
                    for (int k = 0; k < n; ++k)
                        printf("%d ", perm[k]);
                    printf("\n");
                }
            }
        }
    } 

    // Add up all the subtotals to get total number of solutions for the nqueens
    // problem
    MPI_Reduce(&subtotal, &grand_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM , 0, 
        MPI_COMM_WORLD);
    
    MPI_Barrier (MPI_COMM_WORLD);
    /* Stop timer */
    elapsed_time += MPI_Wtime(); /* elapsed time=current time-start time */

    if (0 == id) {
        printf ("%lld\n", grand_total ); 
        printf ("Execution time %8.3f ms\n", 1000*elapsed_time);
        fflush (stdout);
    }

    MPI_Finalize();
    
    free(perm);
    
    return 0;
}
