#include <stdlib.h>
#include <stdio.h> 
#include <malloc.h>
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/math_libs/11.5/targets/x86_64-linux/include/cublas_v2.h"
#include </opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda/11.0/targets/x86_64-linux/include/cuda_runtime.h>


int main(int argc, char *argv[]){
    //size of matrix, tolerance, iter_max
    int size = strtol(argv[1], NULL, 10);

    float tol = atof(argv[2]);
    int iter_max = strtol(argv[3], NULL, 10);

    float* A = (float*)malloc(((size + 2)*(size+2)) * sizeof(float));
    float* Anew = (float*)malloc(((size + 2)*(size+2)) * sizeof(float));

    int iter = 0;
    float error = 100.0;
    float *ptr;
    float add_gradient = 10.0 / (size + 2); 
    
    #pragma acc enter data create(A[0:((size+2)*(size+2))], Anew[0:((size+2)*(size+2))]) copyin(size, add_gradient)

    #pragma acc data present(A, Anew)
    #pragma acc kernels
    {
    for (int i = 0; i < size + 2; i++){
        A[i*(size+2) + 0] = 10 + add_gradient*i;
        A[i] = 10 + add_gradient*i;
        A[size+1 + i] = 20 + add_gradient*i;
        A[i*(size+2)+size+1] = 20 + add_gradient*i;

        Anew[i*(size+2) + 0] = A[i*(size+2) + 0];
        Anew[i] = A[i];
        Anew[size+1 + i] = A[size+1 + i];
        Anew[i*(size+2)+size+1] = A[i*(size+2)+size+1];
    }
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    int index;
    float max;
    float neg_one = (-1.0);

    while ((error > tol) && (iter < iter_max)){

        iter++;

        #pragma acc data present(A, Anew)
        #pragma acc kernels async(1)
        {
        #pragma acc loop independent collapse(2)
        for (int j = 1; j < size + 1; j++){
            for (int i = 1; i < size + 1; i++){
                Anew[i*(size+2)+j] = 0.25 * (A[(i+1)*(size+2)+j] + A[(i-1)*(size+2)+j] + A[i*(size+2)+j-1] + A[i*(size+2)+j+1]);
            }
        }
        }

        
        if ((iter % 150 == 0) || (iter == 1)){
            #pragma acc wait(1)
            #pragma acc host_data use_device(A, Anew) 
            {
                cublasSaxpy(handle, ((size+2)*(size+2)), &neg_one, Anew, 1, A, 1);
                cublasIsamax(handle, ((size+2)*(size+2)), A, 1, &index);
            }

            #pragma acc update self(A[index-1:1])
            max = A[index-1];
            max = (max >= 0) ? max : -max;

            #pragma acc host_data use_device(A, Anew)
            {
            cublasScopy(handle, ((size+2)*(size+2)), Anew, 1, A, 1);
            }

            if(error > max)
                error = max;

            printf("%d : %lf\n", iter, error);
        }

        ptr = A;
        A = Anew;
        Anew = ptr;
    }

    printf("%d : %lf\n", iter, error);

    #pragma acc exit data delete(A[0:((size+2)*(size+2))], Anew[0:((size+2)*(size+2))])

    free(A);
    free(Anew);

    return 0;
}
