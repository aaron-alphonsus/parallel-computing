/* File:                                                                         
 *    prog1.c                                                                    
 *                                                                               
 * Purpose:                                                                      
 *    A brute force solution to a circuit-satisfiability question parallelizing
 *    the algorithm using OpenMP. Compares time taken by parallel and serial 
 *    approaches.     
 *                                                                               
 * Input:                                                                        
 *    none (pre-defined circuit in the form of an if statement)                            
 * Output:                                                                       
 *    All combinations of inputs that satisfy the circuit.                                      
 *                                                                               
 * Compile:                                                                      
 *    gcc -g -Wall -fopenmp -o prog1 prog1.c                                 
 *    OR make prog1                                                              
 * Usage:                                                                        
 *    ./prog1 <print> <reps>                                                 
 *                                                                               
 * Professor:                                                                    
 *    Dr. Christer Karlsson                                                      
 * Authors:                                                                      
 *    Aaron Alphonsus                                                            
 * Class:                                                                        
 *    CSC410 - Parallel Computing                                                
 */

// Return 1 if 'i'th bit of 'n' is 1; 0 otherwise 
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)

#include <omp.h>
#include <math.h>
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
   fprintf(stderr, "Usage: %s <print> <reps>\n", prog_name); 
   fprintf(stderr, "    Enter 1 or 0 to print or suppress printing.\n");
   fprintf(stderr, "    Number of repititions: reps >= 1"
        " (for timing purposes).\n");
   
    exit(0);
}

/*------------------------------------------------------------------------------
 * Function: check_circuit
 *
 * Purpose:  Check if a given input produces TRUE (a one)
 *
 * In args:  id, z, print
 *
 * Returns:  1: Circuit produces true for given input 
 *           0: Circuit produces false
 */
int check_circuit(int id, int z, int print)
{
    int v[16]; // Each element is a bit of z
    int i;
   
    // Convert z to binary and store each bit in an element of v 
    for (i = 0; i < 16; i++) 
        v[i] = EXTRACT_BIT(z,i);
   
    // If statement representing the circuit we are given 
    if (((v[0] || v[1]) && (!v[1] || !v[3]) && (v[2] || v[3])
        && (!v[3] || !v[4]) && (v[4] || !v[5])
        && (v[5] || !v[6]) && (v[5] || v[6])
        && (v[6] || !v[15]) && (v[7] || !v[8])
        && (!v[7] || !v[13]) && (v[8] || v[9])
        && (v[8] || !v[9]) && (!v[9] || !v[10])
        && (v[9] || v[11]) && (v[10] || v[11])
        && (v[12] || v[13]) && (v[13] || !v[14])
        && (v[14] || v[15])) 
        && print) 
    {
        // print id of process that calls the function and the input combination
        printf ("%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", id,
            v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],
            v[13],v[14],v[15]);
        
        fflush (stdout);
        return 1;
    } 
    else 
        return 0;
}

/*------------------------------------------------------------------------------
 * Function: time_parallel
 *
 * Purpose:  Runs algorithm to check the circuit satisfiability reps times
 *           Keeps track of how long the algorithm takes
 *           Evaluates output for each input combination in parallel
 *
 * In args:  inputs, thread_count, reps, print
 *
 * Out args: solutions, time
 *
 * Returns:  time/reps: Time to run the algorithm averaged over reps times
 */
double time_parallel(int inputs, int thread_count, int reps, int print)
{
    int i, j, solutions = 0;

    double begin, end;
    double time = 0.0;

    // run algorithm reps times for timing purposes
    for(i = 0; i < reps; i++)                                                           
    {                                                                             
        begin = omp_get_wtime(); 
       
        // Test each of the 65536 outputs in parallel 
        # pragma omp parallel for num_threads(thread_count) \
            default(none) reduction(+: solutions) private(i) \
            shared(inputs, print) // schedule(dynamic, 1) 
        for(j = 0; j < inputs; j++)
        {    
            int id = omp_get_thread_num();
            solutions += check_circuit(id, j, print);
        }
        
        end = omp_get_wtime();
        time += (double)((end-begin)*1000.0);
    }
    // print number of solutions if print requested
    if(print)
        printf("Number of solutions = %d\n\n", solutions);    

    return time/reps;
}

/*------------------------------------------------------------------------------
 * Function: time_serial
 *
 * Purpose:  Runs algorithm to check the circuit satisfiability reps times
 *           Keeps track of how long the algorithm takes
 *           Evaluates output for each input combination serlially
 *
 * In args:  inputs, reps, print
 *
 * Out args: solutions, time
 *
 * Returns:  time/reps: Time to run the algorithm averaged over reps times
 */
double time_serial(int inputs, int reps, int print)
{
    int i, j, solutions = 0;
    
    double begin, end;
    double time = 0.0; 

    // run algorithm reps times for timing purposes
    for(i = 0; i < reps; i++)                                                           
    {                                                                             
        begin = omp_get_wtime(); 
        
        // Test each of the 65536 outputs serially
        for(j = 0; j < inputs; j++) 
            solutions += check_circuit(j, j, print);    
        
        end = omp_get_wtime();
        time += (double)((end-begin)*1000.0);
    }
    // print number of solutions if print requested
    if(print)
        printf("Number of solutions = %d\n\n", solutions);
    
    return time/reps; 
}

/*------------------------------------------------------------------------------
 * Function: main
 *
 * Purpose:  Checks command-line arguments and calls usage function if invalid
 *           Calls parallel and serial functions run and time each method 
 *           Prints time taken by each method
 *
 * In args:  argc, argv
 *
 * Out args: time_par, time_ser
 *
 * Returns:  0 indicating normal termination
 */
int main(int argc, char* argv[])
{
    int print, reps;

    // Check if command line arguments are valid
    if(argc != 3) 
        usage(argv[0]);
 
    print = strtol(argv[1], NULL, 10);
    reps = strtol(argv[2], NULL, 10);
 
    if(print != 1 && print != 0)
        usage(argv[0]);
    if(reps < 1)
        usage(argv[0]);

    // Declare variables and call functions to time each algorithm
    int inputs = pow(2, 16);    
    int thread_count = 8; 
     
    double time_par, time_ser;
   
    time_par = time_parallel(inputs, thread_count, reps, print);
    time_ser = time_serial(inputs, reps, print);     

    printf("Parallel time = %.4lf ms\n", time_par);
    printf("Serial time = %.4lf ms\n", time_ser); 
 
    return 0;
}
