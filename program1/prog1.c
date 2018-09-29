/* Return 1 if 'i'th bit of 'n' is 1; 0 otherwise */
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)

#include <omp.h>
#include <math.h>
#include <stdio.h>

/* Check if a given input produces TRUE (a one) */
int check_circuit (int id, int z)
{
    int v[16]; /* Each element is a bit of z */
    int i;
    
    for (i = 0; i < 16; i++) 
        v[i] = EXTRACT_BIT(z,i);
    
    if ((v[0] || v[1]) && (!v[1] || !v[3]) && (v[2] || v[3])
        && (!v[3] || !v[4]) && (v[4] || !v[5])
        && (v[5] || !v[6]) && (v[5] || v[6])
        && (v[6] || !v[15]) && (v[7] || !v[8])
        && (!v[7] || !v[13]) && (v[8] || v[9])
        && (v[8] || !v[9]) && (!v[9] || !v[10])
        && (v[9] || v[11]) && (v[10] || v[11])
        && (v[12] || v[13]) && (v[13] || !v[14])
        && (v[14] || v[15])) 
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
    int n = pow(2, 16);    
    int thread_count = 8; 

    # pragma omp parallel for num_threads(thread_count)
    for(int i = 0; i < n; i++)
    {    
        int id = omp_get_thread_num();
        // printf("%d: %d\n", i, check_circuit(i, i));
        // printf("%d\n", id);
        check_circuit(id, i);
    }

    return 0;
}
