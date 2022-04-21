#include <stdlib.h>
#include <stdio.h> 
#include <math.h>
#include <malloc.h>
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/math_libs/11.5/targets/x86_64-linux/include/cublas_v2.h"
#include </opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda/11.0/targets/x86_64-linux/include/cuda_runtime.h>
#include <cub/cub.cuh>
//for(i=0 i<1024)
//A[i] = smth
//разбивает на 8 векторов длиной 128
//foo<<<8, 128>>> 128<1024
//https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf


#define CHKCUDA(err) if(err != cudaSuccess) { fprintf(stderr, "Error%s\n", cudaGetErrorString(err)); exit(1); }
#define max(x, y) ((x) > (y) ? (x) : (y) )
#define THREADS_PER_BLOCK 32
//cuda-memcheck ./cuda.out
//addr2line -e ./cuda.out  -a 0x12e4b 0x6cbc8 0xb125 0xa5f5 0xa651

__global__ void calc(double* A, double* Anew, int n){
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < n - 1 && j > 0 && i > 0 && i < n - 1){
		Anew[i*n + j] = 0.25 * (A[i*n + j - 1] + A[i*n + j + 1] + A[(i-1)*n + j] + A[(i+1)*n + j]);    
	}
}

__global__ void arr_diff(double* A, double* Anew, double* max, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if((i > 1) && (j > 1) && (i < (size+1)) && (j < (size+1))){
        max[i*(size+2)+j] = Anew[i*(size+2)+j] - A[i*(size+2)+j];
    }
}


__global__ void init(double *A, double *Anew, int size, double gradient){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < (size+2)){
        A[i*(size+2) + 0] = 10 + gradient*i;
        A[i] = 10 + gradient*i;
        A[(size+1)*(size+2) + i] = 20 + gradient*i;
        A[i*(size+2)+size+1] = 20 + gradient*i;

        Anew[i*(size+2) + 0] = A[i*(size+2) + 0];
        Anew[i] = A[i];
        Anew[(size+1)*(size+2) + i] = A[(size+1)*(size+2) + i];
        Anew[i*(size+2)+size+1] = A[i*(size+2)+size+1];
    }
}


int main(int argc, char *argv[]){
    //size of matrix, tolerance, iter_max
    int size = strtol(argv[1], NULL, 10);
    double tol = atof(argv[2]);
    int iter_max = strtol(argv[3], NULL, 10);

    cudaError_t err;
    double* A = (double*)malloc(((size+2)*(size+2)) * sizeof(double));
    double* A_d;
    double* Anew = (double*)malloc(((size+2)*(size+2)) * sizeof(double));
    double* Anew_d;
    double* max_array = (double*)malloc(((size+2)*(size+2)) * sizeof(double));
    double* max_array_d;
    double* max_number_d;

    err = cudaMalloc(&A_d, ((size+2)*(size+2))*sizeof(double));
    CHKCUDA(err)
    err = cudaMalloc(&Anew_d, ((size+2)*(size+2))*sizeof(double));
    CHKCUDA(err)
    err = cudaMalloc(&max_array_d, ((size+2)*(size+2))*sizeof(double));
    CHKCUDA(err)
    err = cudaMalloc(&max_number_d, sizeof(double));
    CHKCUDA(err)

    int iter = 0;
    double error = 100.0;
    double *ptr;
    double add_gradient = 10.0 / (size + 1); 


    //dim3 BS = 544;
    dim3 BS = dim3(THREADS_PER_BLOCK);
    //dim3 GS = 32;
    dim3 GS = dim3(int(ceil((((size+2)*(size+2))/(double)THREADS_PER_BLOCK))));

    init<<<GS, BS>>>(A_d, Anew_d, size, add_gradient);

    //BS = dim3(32, 32);
    BS = dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    int gsx = int(ceil((size+2)/(double)THREADS_PER_BLOCK));
    //GS = dim3(5, 5);
    GS = dim3(gsx, gsx);

    //https://developer.ridgerun.com/wiki/index.php?title=How_to_use_OpenCV_CUDA_Streams
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, max_array_d, max_number_d, ((size+2)*(size+2)));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    while ((error > tol) && (iter < iter_max)){

        iter++;
        calc<<<GS,BS, 0, stream1>>>(A_d, Anew_d, size+2);

        if ((iter % 150 == 0) || (iter == 1)){
            
            cudaDeviceSynchronize();

            error = 0.0;

            arr_diff<<<GS, BS>>>(A_d, Anew_d, max_array_d, size);
            
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, max_array_d, max_number_d, ((size+2)*(size+2)));
            cudaMemcpy(&error, max_number_d, sizeof(double), cudaMemcpyDeviceToHost);

            printf("%d : %lf\n", iter, error);
        }

        ptr = A_d;
        A_d = Anew_d;
        Anew_d = ptr;
    }

    printf("%d : %lf\n", iter, error);

    return 0;
}