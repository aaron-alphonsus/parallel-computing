/* File:                                                                         
 *    prog2.c                                                                    
 *                                                                               
 * Purpose:                                                                      
 *    TODO A brute force solution to a circuit-satisfiability question parallelizing
 *    the algorithm using OpenMP. Compares time taken by parallel and serial 
 *    approaches.     
 *                                                                               
 * Input:                                                                        
 *    TODO none (pre-defined circuit in the form of an if statement)                            
 * Output:                                                                       
 *    TODO All combinations of inputs that satisfy the circuit.
 *                                                                               
 * Compile: TODO                                                                     
 *    gcc -g -Wall -fopenmp -o prog1 prog1.c                                 
 *     OR 
 *    make prog1                                                              
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

#include <stdio.h>
// #include <stdlib.h>
// #include <math.h>

void matvecMul_serial(unsigned long long int *A, unsigned long long int *B, 
unsigned long long int *C, int n)
{
    unsigned long long int sum = 0;

    for(int i = 0; i < n; i++)
    {
        sum = 0;
        for (int j = 0; j < n; j++)
            sum += A[i * n + j] * B[j];  
        C[i] = sum;
    }
}

__global__ void matvecMul(unsigned long long int *A, unsigned long long int *B, 
unsigned long long int *C, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int sum = 0;
   
    if(index < n)
    { 
        // printf("index, A[index] = %d, %d\n", index, A[index]);
        for(int j = 0; j < n; j++)
            sum += A[index * n + j] * B[j];
        C[index] = sum;
    }
}

__global__ void matvecMul_flex(unsigned long long int *A, 
unsigned long long int *B, unsigned long long int *C, int n)
{
    unsigned long long int sum = 0;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = index; i < n; i += stride)
    {
        sum = 0; 
        for(int j = 0; j < n; j++)
            sum += A[i * n + j] * B[j];
        C[i] = sum;
    }
}

int main(int argc, char* argv[])
{ 
    // Size of vectors
    int n = 4096;
    unsigned long long int sumofsq = 0; 
    unsigned long long int wrong = 0;

    // Device input vectors
    unsigned long long int *A;
    unsigned long long int *B;
    // Device output vectors
    unsigned long long int *C;

    // Size, in bytes, of 'A' "matrix"
    size_t mat_bytes = n * n * sizeof(double); 
    // Size, in bytes, of 'B' and 'C' vectors
    size_t vec_bytes = n * sizeof(double);

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&A, mat_bytes);
    cudaMallocManaged(&B, vec_bytes);
    cudaMallocManaged(&C, vec_bytes);

    // Initialize vectors on host
    for(int i = 0; i < n; i++) 
        for(int j = 0; j < n; j++)
            A[i * n + j] = j; 
    for(int i = 0; i < n; i++) 
        B[i] = i;

    // int padding = 3;
    // printf("Matrix A\n");
    // for(int i = 0; i < n; i++) 
    // {
    //     for(int j = 0; j < n; j++)
    //         printf("%*lld", padding, A[i * n + j]);
    //     printf("\n");
    // }

    // printf("Matrix B\n");
    // for(int i = 0; i < n; i++) 
    //     printf("%lld ", B[i]);
    // printf("\n");

    // No. of threads in each thread block and no. of thread blocks in grid
    int blockSize = 256;
    int gridSize = (int)ceil((float)n/blockSize);
    // Execute the kernel
    matvecMul_flex<<<gridSize, blockSize>>>(A, B, C, n); 
   
    // matvecMul_serial(A, B, C, n);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();
  
    for(int i = 0; i < n; i++)
        sumofsq += i*i;
    
    printf("Matrix C\n");
    for(int i = 0; i < n; i++) 
        printf("%lld ", C[i]);
    printf("\n");
 
    // Check each C element with sumofsq
    for(int i = 0; i < n; i++) 
        if(sumofsq != C[i])
            wrong++;
    printf("Number of positions incorrect = %lld\n", wrong); 

    printf("n, sumofsq = %d, %lld\n", n, sumofsq);
    printf("<<<gridSize, blockSize>>> = <<<%d, %d>>>\n", gridSize, blockSize);

    // Release Unified Memory 
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
