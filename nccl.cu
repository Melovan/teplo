#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>
#include <nccl.h>

#define NUM_DEVICES 4

#define CUDACHKERR(err) if (err != cudaSuccess) { \
    fprintf(stderr, \
            "Failed (error code %s)!\n", \
            cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  }

void setDevice(int rank)
{
    cudaError_t err;

    int num_devices = 0;
    err = cudaGetDeviceCount(&num_devices);
    CUDACHKERR(err);

    if (NUM_DEVICES > num_devices)
    {
        fprintf(stderr, "Error: number of devices (%d) is less then %d\n", num_devices, NUM_DEVICES);
        exit(-1);
    }
    cudaSetDevice(rank % NUM_DEVICES);
}

// [start, end)
void getArrayBoundaries(int* start, int* end, int rank, int mx_size, int rank_count)
{
    int y_size = mx_size / rank_count;
    int remains = mx_size - y_size * rank_count;

    int *y_starts = (int*)calloc(rank_count, sizeof(int));

    y_starts[0] = 0;
    for (int i = 1; i < rank_count; ++i)
    {
        y_starts[i] = y_starts[i - 1] + y_size;
        if (remains > 0)
        {
            y_starts[i] += 1;
            --remains;
        }
    }
    
    int *y_ends = (int*)calloc(rank_count, sizeof(int));

    y_ends[rank_count - 1] = mx_size;
    if (rank != rank_count - 1)
    {
        y_ends[rank] = y_starts[rank + 1];
    }

    *start = y_starts[rank] * mx_size;
    *end = y_ends[rank] * mx_size;

    free(y_starts);
    free(y_ends);
}

void interpolateHorizontal(double* arr, double leftValue, double rightValue, int startPosition, int mx_size)
{
    arr[startPosition] = leftValue;
    arr[startPosition + mx_size - 1] = rightValue;

    double step = (rightValue - leftValue) / ((double)mx_size - 1);
    for (int i = startPosition + 1; i < startPosition + mx_size - 1; ++i)
    {
        arr[i] = arr[i - 1] + step;
    }
}

void interpolateVertical(double* arr, double topValue, double bottomValue, int startPos, int yPos, int numRows, int mx_size)
{
    for (int i = 0; i < numRows; ++i)
    {
        arr[i * mx_size + startPos] = (topValue * (mx_size - 1 - i - yPos) + bottomValue * (i + yPos)) / (mx_size - 1);
    }
}

double* getSetMatrix(double* dst, int numElems, int matrix_size, cudaStream_t stream)
{
    cudaError_t err;

    double *matrix;
    err = cudaMalloc((void **)&matrix, (numElems + 2 * matrix_size) * sizeof(double));
    CUDACHKERR(err);

    // filling with zeros cuda matrix
    double *zeroMx = (double*)calloc(numElems + 2 * matrix_size, sizeof(double));
    err = cudaMemcpyAsync(matrix, zeroMx, (numElems + 2 * matrix_size) * sizeof(double), cudaMemcpyHostToDevice, stream);
    CUDACHKERR(err);

    // copying the matrix from the CPU to the GPU, with the space for the boundary values
    err = cudaMemcpyAsync(matrix + matrix_size, dst, numElems * sizeof(double), cudaMemcpyHostToDevice, stream);
    CUDACHKERR(err);

    free(zeroMx);
    return matrix;
}

__global__ void evalEquation(double *newA, const double *A, int width, int y_start, int y_end)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < idx && idx < width - 1) && (y_start < idy && idy < y_end))
    {
        newA[idy * width + idx] = 0.25 * (A[(idy - 1) * width + idx] + A[(idy + 1) * width + idx] +
                                          A[idy * width + (idx - 1)] + A[idy * width + (idx + 1)]);
    }
}

__global__ void vecNeg(const double *newA, const double *A, double* ans, int mx_size, int numElems)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= mx_size && idx < numElems + mx_size)
    {
        ans[idx] = newA[idx] - A[idx];
    }
}

int main(int argc, char *argv[])
{
    int local_rank, numProcess;
    MPI_Init(&argc, &argv);

    double min_error = atof(argv[1]);
    int matrix_size = atoi(argv[2]);
    int iter_max = atoi(argv[3]);

    MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);

    ncclUniqueId id;
    if (local_rank == 0) ncclGetUniqueId(&id);

    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    setDevice(local_rank);

    cudaError_t err;

    double *tmp = NULL;
    double *A_d = NULL;
    double *newA_d = NULL;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    double *tmp_d = NULL;
    double *max_d = NULL;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int start, end;
    getArrayBoundaries(&start, &end, local_rank, matrix_size, numProcess);

    int numElems = end - start;
    int numRows = numElems / matrix_size;

    // interpolation for different processes
    tmp = (double*)calloc(numElems, sizeof(double));

    if (local_rank == 0)
    {
        interpolateHorizontal(tmp, 10.0, 20.0, 0, matrix_size);
    }
    if (local_rank == numProcess - 1)
    {
        interpolateHorizontal(tmp, 20.0, 30.0, numElems - matrix_size, matrix_size);
    }
    interpolateVertical(tmp, 10.0, 20.0, 0, start / matrix_size, numRows, matrix_size);
    interpolateVertical(tmp, 20.0, 30.0, matrix_size - 1, start / matrix_size, numRows, matrix_size);

    // copying to GPU

    A_d = getSetMatrix(tmp, numElems, matrix_size, stream);
    newA_d = getSetMatrix(tmp, numElems, matrix_size, stream);
    free(tmp);

    dim3 GS = dim3(16, 16);
    dim3 BS = dim3(ceil(matrix_size / (double)GS.x), ceil((numRows + 2) / (double)GS.y));

    cudaMalloc(&tmp_d, sizeof(double) * numElems);
    cudaMalloc(&max_d, sizeof(double));

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, numRows * matrix_size, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // calculation of ranks of processes with which boundary values are exchanged
    int topProcess = local_rank != numProcess - 1 ? local_rank + 1 : 0;
    int bottomProcess = local_rank != 0 ? local_rank - 1 : numProcess - 1;

    // calculation of values that limit the upper and lower boundaries for processes, depending on the position
    int y_start = local_rank == 0 ? 1 : 0;
    int y_end = local_rank == numProcess - 1 ? numRows : numRows + 1;

    int GS_neg = matrix_size;
    int BS_neg = ceil(numElems / (double)GS_neg);

    int iter = 0;
    double error = 12321;
    double *gpu_error;
    cudaMalloc(&gpu_error, sizeof(double));

    ncclComm_t comm;
    ncclCommInitRank(&comm, numProcess, id, local_rank);

    while (error > min_error && iter < iter_max)
    {
        ++iter;

        ncclGroupStart();
        ncclSend(A_d + matrix_size, matrix_size, ncclDouble, bottomProcess, comm, stream);
        ncclSend(A_d + numElems, matrix_size, ncclDouble, topProcess, comm, stream);

        ncclRecv(A_d + numElems + matrix_size, matrix_size, ncclDouble, topProcess, comm, stream);
        ncclRecv(A_d, matrix_size, ncclDouble, bottomProcess, comm, stream);
        ncclGroupEnd();

        evalEquation<<<BS, GS, 0, stream>>>(newA_d, A_d, matrix_size, y_start, y_end);

        if (iter % 100 == 0)
        {       
            vecNeg<<<BS_neg, GS_neg, 0, stream>>>(newA_d, A_d, tmp_d, matrix_size, numElems);

            err = cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, numRows * matrix_size, stream);
            CUDACHKERR(err);

            ncclAllReduce(max_d, gpu_error, 1, ncclDouble, ncclMax, comm, stream);

            err = cudaMemcpyAsync(&error, gpu_error, sizeof(double), cudaMemcpyDeviceToHost, stream);
            CUDACHKERR(err);

            if (local_rank == 0)
            {
                printf("iter = %d error = %e\n", iter, error);                
            }
        }
        
        double *tmp = A_d;
        A_d = newA_d;
        newA_d = tmp;
    }

    cudaFree(A_d);
    cudaFree(newA_d);
    cudaFree(tmp_d);
    cudaFree(max_d);

    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    MPI_Finalize();

    return 0;
}
