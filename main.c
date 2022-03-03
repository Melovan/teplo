#include <stdlib.h>
#include <stdio.h> 
#include <malloc.h>
#define size 128
#define tol 0.000001
#define iter_max 10000
#define max(x, y) ((x) > (y) ? (x) : (y) )

int main(int argc, char* argv[]) {
    //size of matrix, tolerance, iter_max


    double** A = (double**)malloc((size + 2) * sizeof(double*));
    double** Anew = (double**)malloc((size + 2) * sizeof(double*));
    for (int i = 0; i < size + 2; ++i) {
        A[i] = (double*)malloc((size + 2) * sizeof(double));
        Anew[i] = (double*)malloc((size + 2) * sizeof(double));
    }

    int iter = 0;
    double error = 1.0;
    double** ptr;
    double add_gradient = 10.0 / (size + 2);

#pragma acc enter data create(A[0:size+2][0:size+2], Anew[0:size+2][0:size+2]) copyin(size, add_gradient)

#pragma acc kernels
    {
        for (int i = 0; i < size + 2; i++) {
            A[i][0] = 10 + add_gradient * i;
            A[0][i] = 10 + add_gradient * i;
            A[size + 1][i] = 20 + add_gradient * i;
            A[i][size + 1] = 20 + add_gradient * i;

            Anew[i][0] = A[i][0];
            Anew[0][i] = A[0][i];
            Anew[size + 1][i] = A[size + 1][i];
            Anew[i][size + 1] = A[i][size + 1];
        }
    }


#pragma acc data create(error)
    {
        while ((error > tol) && (iter < iter_max)) {

            iter++;

            if ((iter % 150 == 0) || (iter == 1)) {
#pragma acc kernels async(1)
                error = 0.0;

#pragma acc data present(A, Anew)
#pragma acc kernels async(1)
                {
#pragma acc loop independent collapse(2) reduction(max:error)
                    for (int j = 1; j < size + 1; j++) {
                        for (int i = 1; i < size + 1; i++) {
                            Anew[i][j] = 0.25 * (A[i + 1][j] + A[i - 1][j] + A[i][j - 1] + A[i][j + 1]);
                            error = max(error, Anew[i][j] - A[i][j]);
                        }
                    }
                }

            }
            else {

#pragma acc data present(A, Anew)
#pragma acc kernels async(1)
                {
#pragma acc loop independent collapse(2)
                    for (int j = 1; j < size + 1; j++) {
                        for (int i = 1; i < size + 1; i++) {
                            Anew[i][j] = 0.25 * (A[i + 1][j] + A[i - 1][j] + A[i][j - 1] + A[i][j + 1]);
                        }
                    }
                }
            }

            ptr = A;
            A = Anew;
            Anew = ptr;

            if ((iter % 150 == 0) || (iter == 1)) {
#pragma acc wait(1)
#pragma acc update host(error)
                printf("%d : %lf\n", iter, error);
            }
        }
    }
    printf("%d : %lf\n", iter, error);


    return 0;
}