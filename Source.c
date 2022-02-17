#include <stdio.h>
#include <math.h>
#define N  128
#define tau (double)1 / (2*N*N)
#define H  (double)1 / N

int main()
{
	double array[N][N] = { 0 }, arrayn[N][N] = { 0 };
	array[N - 1][0] = 30;
	array[N - 1][N - 1] = 20;
	array[0][0] = 10;
	array[0][N-1] = 20;
	arrayn[N - 1][0] = 30;
	arrayn[N - 1][N - 1] = 20;
	arrayn[0][0] = 10;
	arrayn[0][N - 1] = 20;
	int iter = 0;
	double step1 = (array[0][0] - array[N - 1][0]) / N;
	double step2 = (array[N-1][0] - array[N - 1][N-1]) / N;
	double step3 = (array[0][N-1] - array[0][0]) / N;
	double step4 = 20;
	for (int i = 1; i < N-1; i++)
	{
		array[i][0] = array[i - 1][0] + step1;
		array[0][i] = array[0][i - 1] + step3;
		array[N - 1][i] = array[N - 1][i - 1] + step2;
		array[i][N - 1] = step4;

	}
	for (int i = 0;i < N;i++)
	{
		for (int j = 0;j < N;j++)
		{
			printf("%lf ", array[i][j]);
		}
		printf("\n");
	}
	return 0;
}