#include <iostream>
#include <Eigen/Dense>
#include <cblas.h>

using namespace Eigen;

#define M 4000
#define N 200
#define K 3000

void integer_time(){
    MatrixXi a = (MatrixXd::Random(M, K) * 1024).cast<int>();
    MatrixXi b = (MatrixXd::Random(K, N) * 1024).cast<int>();
    MatrixXi d(M, N);

    clock_t begin = clock();
    d = a*b;
    clock_t end = clock();
    double sec = double(end- begin) / CLOCKS_PER_SEC;
    std::cout << "integer time(eigen): " << sec << std::endl;

    const float alpha=1;
    const float beta=0;
    float *a_arr = (float *)malloc(M*K*sizeof(float));
    float *b_arr = (float *)malloc(K*N*sizeof(float));
    float *c_arr = (float *)malloc(M*N*sizeof(float));

    for (int i = 0; i < M; i++)
    {
       for (int j = 0; j < K; j++)
       {
           a_arr[i * K + j] = a(i, j);
       }
    }

    for (int i = 0; i < K; i++)
    {
       for (int j = 0; j < N; j++)
       {
           b_arr[i * N + j] = b(i, j);
       }
    }

    begin = clock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a_arr, K, b_arr, N, beta, c_arr, N);
    end = clock();
    sec = double(end- begin) /CLOCKS_PER_SEC;
    std::cout << "integer time(blas): " << sec << std::endl;
    free(a_arr);
    free(b_arr);
    free(c_arr);
}

void double_time(){
    MatrixXd a = MatrixXd::Random(M, K);
    MatrixXd b = MatrixXd::Random(K, N);
    MatrixXd d(M, N);

    clock_t begin = clock();
    d = a*b;
    clock_t end = clock();
    double sec = double(end- begin) /CLOCKS_PER_SEC;
    std::cout << "double time(eigne): " << sec << std::endl;

    const float alpha=1;
    const float beta=0;
    double *a_arr = (double *)malloc(M*K*sizeof(double));
    double *b_arr = (double *)malloc(K*N*sizeof(double));
    double *c_arr = (double *)malloc(M*N*sizeof(double));

    for (int i = 0; i < M; i++)
    {
       for (int j = 0; j < K; j++)
       {
           a_arr[i * K + j] = a(i, j);
       }
    }

    for (int i = 0; i < K; i++)
    {
       for (int j = 0; j < N; j++)
       {
           b_arr[i * N + j] = b(i, j);
       }
    }

    begin = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a_arr, K, b_arr, N, beta, c_arr, N);
    end = clock();
    sec = double(end- begin) / CLOCKS_PER_SEC;
    std::cout << "double time(blas): " << sec << std::endl;
    free(a_arr);
    free(b_arr);
    free(c_arr);
}

int main()
{
    integer_time();
    double_time();
    return 0;
}

