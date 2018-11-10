// Sequential program to sum up two vectors
// Compile: gcc -g -Wall -o vecAdd vecAdd.c -lm
// Author:  Dr. Christer Karlsson

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
// function to add the elements of two arrays
void vecAdd(double *a, double *b, int n)
{
    for(int i = 0; i < n; i++)
       b[i] = a[i] + b[i];
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 1<<20;
    printf("%d\n", n);
        
    // Device input vectors
    double *d_a;
    double *d_b;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate Memory
    d_a = (double*)malloc(bytes);
    d_b = (double*)malloc(bytes);
 
    // Initialize vectors
    for(int i = 0; i < n; i++ ) 
    {
        d_a[i] = sin(i)*sin(i);
        d_b[i] = cos(i)*cos(i);
    }
 
    // do the addition
    vecAdd(d_a, d_b, n);
 
    /* Sum up vector d_b and print result divided by n, 
       this should equal 1 within error (sin^2(x) + cos^2(x) = 1 */ 
    double sum = 0.0;
    for(int i=0; i<n; i++)
        sum += d_b[i];
    printf("final result: %f\n", sum/n);
 
    // Free the allocated memory 
    free(d_a);
    free(d_b);
 
    return 0;
}
