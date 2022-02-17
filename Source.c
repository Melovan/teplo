#include <stdio.h>
#include <math.h>
#define N  128
int main()
{
	double array[N] = { 0 }, arrayn[N] = { 0 };
	array[0] = 0;
	array[N-1] = 1;
	arrayn[0] = 0;
	arrayn[N - 1] = 1;
	double H = (double)1 / N, tau = (double)1 / (2*N*N);
	while(1)
	{
		int c = 0;
		for (int i = 1; i < N - 1;i++)
		{
			
			arrayn[i] = array[i] + tau*(array[i - 1] - 2 * array[i] + array[i + 1]) / (H * H);
			double n = fabs(arrayn[i] - array[i]);
				if (n < 0.000001 )
					c++;
			printf("%lf   %lf   %e\n", arrayn[i], array[i], fabs(arrayn[i] - array[i]));
			array[i] = arrayn[i];
		}
		if (c == 126)
			break;
		printf("\n\n");
	}

	return 0;
}