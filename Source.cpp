#include <iostream>
#include <stdlib.h>
#define accuracy 0.000001
#define size 128
#define iters 1000000
using namespace std;



int main(int argc, char* argv[]) {
    double Q11 = 10.;
    double Q12 = 20.;
    double Q21 = 20.;
    double Q22 = 30.;

    double step = 10. / size;

    double A[size + 2][size + 2] ;

    A[0][0] = Q11;
    A[0][size + 1] = Q12;
    A[size + 1][0] = Q21;
    A[size + 1][size + 1] = Q22;
#pragma acc data copy(A) create(A)
    {
        #pragma acc parallel
        {
            for (int i = 1; i < size + 1; i++) {
                A[0][i] = A[0][i - 1] + step;
                A[size + 1][i] = A[size + 1][i - 1] + step;
            }

            for (int j = 1; j < size + 1; j++) {
                A[j][0] = A[j - 1][0] + step;
                A[j][size + 1] = A[j - 1][size + 1] + step;
            }

            for (int i = 1; i < size + 1; i++)
                for (int j = 1; j < size + 1; j++)
                    A[i][j] = 0;
        }
        
    }
    

    double Anew[size + 2][size + 2] ;
    int iter = 0;
    double err = 1;
#pragma acc data copy(A) create(Anew, err)
    {
        while ((err > accuracy) && (iter < iters)) {
            iter++;

            if ((iter % 100 == 0) || (iter == 1)) {
#pragma acc kernels async(1)
                {
                    err = 0;
#pragma acc loop independent collapse(2) reduction(max:err)
                    for (int j = 1; j < size + 1; j++)
                        for (int i = 1; i < size + 1; i++) {
                            Anew[i][j] = 0.25 * (A[i + 1][j] + A[i - 1][j] + A[i][j - 1] + A[i][j + 1]);
                            err = max(err, Anew[i][j] - A[i][j]);
                        }
                }
            }
            else {
#pragma acc kernels async(1)
#pragma acc loop independent collapse(2)
                for (int j = 1; j < size + 1; j++)
                    for (int i = 1; i < size + 1; i++) {
                        Anew[i][j] = 0.25 * (A[i + 1][j] + A[i - 1][j] + A[i][j - 1] + A[i][j + 1]);

                    }

            }
#pragma acc kernels async(1)
            for (int i = 1; i < size + 1; i++)
                for (int j = 1; j < size + 1; j++)
                    A[i][j] = Anew[i][j];

            if ((iter % 100 == 0) || (iter == 1)) {
#pragma acc wait(1)
#pragma acc update self(err)
                cout << iter << ' ' << err << endl;
            }
        }
    }
   
    

    cout << iter << ' ' << err << endl;
    
    return 0;
}
