#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cub/cub.cuh>
#define N 256
#define MAX_ITER_NUM 15000
#define STEP 100
__global__ void five_point_model_calc(double* U_d, double* U_d_n, int n)
{
	
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < n - 1 && j > 0 && i > 0 && i < n - 1)
		{
			double left = U_d[i*n + j - 1];
			double right = U_d[i*n + j + 1];
			double up = U_d[(i-1)*n + j];
			double down = U_d[(i+1)*n + j];

			U_d_n[i*n + j] = 0.25 * (left + right + up + down);
		}
}

__global__ void arr_diff(double* U_d, double* U_d_n, double* U_d_diff, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= 0 && i < n && j >= 0 && j < n)
		U_d_diff[i*n + j] = U_d_n[i*n + j] - U_d[i*n + j];
}



int main(void)
{

double* U = (double*)calloc(N*N, sizeof(double));
double* U_n =(double*)calloc(N*N, sizeof(double));

double* U_d;
double* U_d_n;
double* U_d_diff;

cudaMalloc(&U_d, sizeof(double)*N*N);
cudaMalloc(&U_d_n, sizeof(double)*N*N);
cudaMalloc(&U_d_diff, sizeof(double)*N*N);
double delta = 10.0 / (N - 1);

for (int i = 0; i < N; i++)
{
	U[i*N] = 10 + delta * i;
	U[i] = 10 + delta * i;
	U[(N-1)*N + i] = 20 + delta * i;
	U[i*N + N - 1] = 20 + delta * i;

	U_n[i*N] = U[i*N];
	U_n[i] = U[i];
	U_n[(N-1)*N + i] = U[(N-1)*N + i];
	U_n[i*N + N - 1] = U[i*N + N - 1];
}


cudaMemcpy(U_d, U, N*N*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(U_d_n, U_n, N*N*sizeof(double), cudaMemcpyHostToDevice);

dim3 BLOCK_SIZE = dim3(32, 32);
dim3 GRID_SIZE = dim3(ceil(N/32.),ceil(N/32.));

int it = 0;

double* err = (double*)calloc(1,sizeof(double));
*err = 1;
double* d_err;
cudaMalloc(&d_err, sizeof(double));

void* d_temp_storage = NULL;
size_t temp_storage_bytes = 0;



cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, U_d_diff, d_err, N*N);
cudaMalloc(&d_temp_storage, temp_storage_bytes);

it = 0;
int max_iters_with_graphs = MAX_ITER_NUM / STEP;

cudaStream_t stream;
cudaStreamCreate(&stream);

bool graphCreated = false;
cudaGraph_t graph;
cudaGraphExec_t instance;

while(*err > 1e-6 && it < max_iters_with_graphs)
{	it += 2;
	if(!graphCreated)
	{
		cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
		for(int i = 0; i < 100; i ++)
		{
			five_point_model_calc<<<GRID_SIZE,BLOCK_SIZE,0,stream>>>(U_d_n, U_d, N);
			five_point_model_calc<<<GRID_SIZE, BLOCK_SIZE,0,stream>>>(U_d, U_d_n, N);
			
			//double* swap_ptr = U_d;
			//U_d = U_d_n;
			//U_d_n = swap_ptr;
		}
		cudaStreamEndCapture(stream, &graph);
		cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

		graphCreated=true;
	}
	cudaGraphLaunch(instance, stream);
	cudaStreamSynchronize(stream);
	
	printf("iter = %d error = %e\n", it*STEP, *err);
	*err = 0;
	double* swap_ptr = U_d;
	U_d = U_d_n;
	U_d_n = swap_ptr;
	
	arr_diff<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(U_d_n, U_d, U_d_diff , N);
	cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, U_d_diff, d_err, N*N, stream);
	cudaMemcpyAsync(err, d_err,sizeof(double),cudaMemcpyDeviceToHost, stream );

	swap_ptr = U_d;
	U_d = U_d_n;
	U_d_n = swap_ptr;


	cudaStreamSynchronize(stream);
}


free(U);
free(U_n);
cudaFree(U_d);
cudaFree(U_d_n);
cudaFree(U_d_diff);
printf("fin = %d %lf\n", it, *err);
return 0;
}
