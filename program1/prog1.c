// TODO: The header of each source file should contain all the normal 
// information (name, class assignment etc.)

/* Return 1 if 'i'th bit of 'n' is 1; 0 otherwise */
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)

#include <omp.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

/* Check if a given input produces TRUE (a one) */
int check_circuit (int id, int z, bool print)
{
    int v[16]; /* Each element is a bit of z */
    int i;
    
    for (i = 0; i < 16; i++) 
        v[i] = EXTRACT_BIT(z,i);
    
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
        printf ("%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", id,
            v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],
            v[13],v[14],v[15]);
        
        fflush (stdout);
        return 1;
    } 
    else 
        return 0;
}

int main()
{
    int i, j;
    int n = pow(2, 16);    
    int thread_count = 8; 
    // printf("%d, %d\n", omp_get_num_procs(), omp_get_max_threads());

    int reps = 100;
    bool print = false;
    
    double begin, end;
    double time = 0.0;

    for(i = 0; i < reps; i++)                                                           
    {                                                                             
        begin = omp_get_wtime(); 
        # pragma omp parallel for num_threads(thread_count) \
            default(none) reduction(+: time) private(i) shared(n, print) \
            schedule(static, 1) 
        for(j = 0; j < n; j++)
        {    
            int id = omp_get_thread_num();
            check_circuit(id, j, print);
            
            // printf("%d: %d\n", j, check_circuit(j, j));
            // printf("%d\n", id);
        }
        end = omp_get_wtime();
        time += (double)((end-begin)*1000.0);
    }
    printf("Time %.4lf ms\n", (double)time / reps);

 
    for(i = 0; i < reps; i++)                                                           
    {                                                                             
        begin = omp_get_wtime(); 
        // # pragma omp parallel for num_threads(thread_count) \
        //     default(none) reduction(+: time) private(i) shared(n, print) \
        //     schedule(static, 1) 
        for(j = 0; j < n; j++)
        {    
            // int id = omp_get_thread_num();
            check_circuit(j, j, print);
            // check_circuit(id, j, print);
            
            // printf("%d: %d\n", j, check_circuit(j, j));
            // printf("%d\n", id);
        }
        end = omp_get_wtime();
        time += (double)((end-begin)*1000.0);
    }
    printf("Time %.4lf ms\n", (double)time / reps); 

    return 0;
}
