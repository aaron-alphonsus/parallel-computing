/*------------------------------------------------------------------------------
 * File:
 *    prog2.cu
 *                                                                               
 * Purpose:                                                                      
 *    Host program and CUDA kernel to multiply a square (n x n) matrix with a
 *    vector of length n. Each row of the square matrix is initialized from 1 to
 *    and the vector is also initialized from 1 to n. The resultant vector
 *    contains the sum of squares from 1 to n.
 *                                                                               
 * Input:                                                                        
 *    none (Matrix and vector initialized to yield sum of squares as result)
 * Output:                                                                       
 *    The matrix multiplication result along with some general information for
 *    debugging purposes
 *                                                                               
 * Compile:
 *    nvcc -o prog2 prog2.cu
 *     OR 
 *    make
 * Usage:                                                                        
 *    ./prog2 (To profile: nvprof ./prog2)
 *                                                                               
 * Professor:                                                                    
 *    Dr. Christer Karlsson                                                      
 * Authors:                                                                      
 *    Aaron Alphonsus                                                            
 * Class:                                                                        
 *    CSC410 - Parallel Computing                                                
 */

#include <stdio.h>

/*------------------------------------------------------------------------------
 * Function: matvecMul
 *
 * Purpose:  CUDA kernel code that executes the matrix-vector multiplication on
 *           the GPU. We have gone with coarse-grained parallelism here: each
 *           thread multiplies a row of A with B serially.
 *
 * In args:  A, B, C, n
 *
 * Out args: C
 *
 * Returns:  void
 */
__global__ void matvecMul(double *A, double *B, double *C, int n)
{
    // Declare variable for dot-product summation
    double sum = 0;
    
    // Threads to index into each row of the square matrix
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // grid-stride

    // A grid-stride loop where each thread multiplies a row of A with B in
    //  parallel
    for(int i = index; i < n; i += stride)
    {
        sum = 0; 

        // Sequential portion: each thread calculates the dot-product summation
        //  serially
        for(int j = 0; j < n; j++)
            sum += A[i * n + j] * B[j];

        C[i] = sum;
    }
}

/*------------------------------------------------------------------------------
 * Function: main
 *
 * Purpose:  Declares vectors and allocates unified memory to them. Initializes
 *           the vector B and each row of the matrix A with numbers from 1 to n.
 *           Calls the CUDA kernel after defining grid and block sizes to
 *           execute the multiplication in parallel. Prints out resultant vector
 *           C.
 *
 * Returns:  0 indicating normal termination
 */
int main()
{ 
    // Size of vectors
    int n = 8192;
    double sumofsq = 0;
    unsigned long long int wrong = 0;

    // Device input vectors
    double *A;
    double *B;
    // Device output vectors
    double *C;

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
            A[i * n + j] = j+1;
    for(int i = 0; i < n; i++) 
        B[i] = i+1;

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
    matvecMul<<<gridSize, blockSize>>>(A, B, C, n);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Calculate sum of squares
    for(int i = 0; i < n; i++)
        sumofsq += (i + 1) * (i + 1);

    // Print out matrix C
    printf("Matrix C\n");
    for(int i = 0; i < n; i++) 
        printf("%0.1lf ", C[i]);
    printf("\n\n");
 
    // Check each C element with sumofsq
    for(int i = 0; i < n; i++) 
        if(sumofsq != C[i])
            wrong++;
    printf("Number of positions incorrect = %lld\n", wrong); 

    printf("n, sumofsq = %d, %0.1lf\n", n, sumofsq);
    printf("<<<gridSize, blockSize>>> = <<<%d, %d>>>\n\n", gridSize, blockSize);

    // Release Unified Memory 
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
