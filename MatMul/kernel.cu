
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <locale>
#include <windows.h>
#include <string>

using namespace std;

void print_matrix(double* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << matrix[i + j] << " ";
        }
        cout << endl;
    }
}

void init_matrix(double* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = rand() / 100;
    }
}

__global__ void mul_matrix_kernel(double* A, double* B, double* C, int n) {
    // Реализация перемножения матриц на GPU CUDA
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    C[n * row + col] = 0;

    if (row < n && col < n) {
        for (int k = 0; k < n; k++) {
            C[n * row + col] += A[row * n + k] * B[k * n + col];
        }
    }
}

void mul_matrix_gpu(double* A, double* B, double* C, int n, float time) {
    dim3 threadsPerBlock(n, n);
    dim3 blocksPerGrid(1, 1);
    if (n * n > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(n) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(n) / double(threadsPerBlock.y));
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    mul_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void mul_matrix_cpu(double* A, double* B, double* C, int n) {
    // Реализация перемножения матриц на CPU
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

bool equals_matrix(double* A, double* B, int n) {
    for (int i = 0; i < n * n; i++) {
        if (A[i] != B[i]) {
            return false;
        }
    }
    return true;
}




int main()
{
    setlocale(LC_ALL, "Russian");

    srand(time(nullptr));

    cout << "Введите количсетво строк и столбцов в матрице: ";
    int n;
    cin >> n;

    size_t bytes = n * n * sizeof(double);

    float time_cpu = 0;
    float time_gpu = 0;

    double *h_A, *h_B, *h_C_cpu, *h_C_gpu;

    cudaMallocHost((void**) &h_A, bytes);
    cudaMallocHost((void**) &h_B, bytes);
    cudaMallocHost((void**) &h_C_gpu, bytes);
    cudaMallocHost((void**) &h_C_cpu, bytes);

    init_matrix(h_A, n);
    init_matrix(h_B, n);

    double* d_A, * d_B, * d_C;

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyDeviceToDevice);

    mul_matrix_gpu(d_A, d_B, d_C, n, time_gpu);

    cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    time_cpu = clock();
    mul_matrix_cpu(h_A, h_B, h_C_cpu, n);
    time_cpu = clock() - time_cpu;

    if (n < 10) {
        cout << "Результат перемножения матрицы на GPU: " << endl;
        print_matrix(h_C_gpu, n);
        cout << "Результат перемножения матрицы на CPU: " << endl;
        print_matrix(h_C_cpu, n);
    }
    else {
        cout << "Результирующие матрицы на GPU и CPU равны? " << equals_matrix(h_C_cpu, h_C_gpu, n) << endl;
    }

    cout << "Время работы программы на GPU (мc): " << time_gpu << endl;
    cout << "Время работы программы на CPU (мc): " << time_cpu << endl;
    cout << "Ускорение " << time_cpu / time_gpu << endl;

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C_gpu);
    cudaFreeHost(h_C_cpu);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    system("pause");
    return 0;
}

