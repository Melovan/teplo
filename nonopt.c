#include <stdlib.h>
#include <stdio.h> 
#include <malloc.h>
#define size 128
#define tol 0.000001
#define iter_max 10
#define max(x, y) ((x) > (y) ? (x) : (y) )


int main(int argc, char* argv[]) {
    //size of matrix, tolerance, iter_max


    double** A = (double**)malloc((size + 2) * sizeof(double*));
    double** Anew = (double**)malloc((size + 2) * sizeof(double*));
    for (int i = 0; i < size + 2; ++i) {
        A[i] = (double*)calloc(size + 2, sizeof(double));
        Anew[i] = (double*)calloc(size + 2, sizeof(double));
    }

    int iter = 0;
    double err = 1.0;

    //Use known point values
    A[0][0] = 10.0;
    A[0][size + 1] = 20.0;
    A[size + 1][0] = 20.0;
    A[size + 1][size + 1] = 30.0;

    double add_gradient = 10.0 / (size + 2);

    for (int i = 1; i < size + 1; i++) {
        A[0][i] = A[0][i - 1] + add_gradient;
        A[size + 1][i] = A[size + 1][i - 1] + add_gradient;
        A[i][0] = A[i - 1][0] + add_gradient;
        A[i][size + 1] = A[i - 1][size + 1] + add_gradient;
    }


    for (int j = 1; j < size + 1; ++j) {
        Anew[0][j] = A[0][j];
        Anew[j][0] = A[j][0];
        Anew[size + 1][j] = A[size + 1][j];
        Anew[j][size + 1] = A[j][size + 1];
    }


#pragma acc data copyin(A[0:size+2][0:size+2], Anew[0:size+2][0:size+2])
    {
        while ((err > tol) && (iter < iter_max)) {

            iter++;
            err = 0.0;

#pragma acc kernels 
            {
#pragma acc loop independent collapse(2) reduction(max:err)  
                for (int j = 1; j < size + 1; j++) {
                    for (int i = 1; i < size + 1; i++) {
                        Anew[i][j] = 0.25 * (A[i + 1][j] + A[i - 1][j] + A[i][j - 1] + A[i][j + 1]);
                        err = max(err, Anew[i][j] - A[i][j]);
                    }
                }
            }

#pragma acc kernels 
            for (int i = 0; i < size + 2; i++)
                for (int j = 0; j < size + 2; j++)
                    A[i][j] = Anew[i][j];

            if ((iter % 100 == 0) || (iter == 1)) {
                printf("%d : %lf\n", iter, err);
            }
        }
    }

    printf("%d : %lf\n\n", iter, err);

    return 0;
}