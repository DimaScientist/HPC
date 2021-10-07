
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <time.h>

using namespace std;

const int threadsPerBlock = 256;


void init_array(double *array, int n) {
	for (int i = 0; i < n; i++) {
		array[i] = rand() / 100.0;
	}
}

void print_array(double *array, int n) {
	for (int i = 0; i < n; i++) {
		if (i < n - 1) {
			cout << array[i] << ", ";
		}
		else {
			cout << array[i] << endl;
		}
	}
}

__global__ void vector_sum_kernel(double *array, int n, double *result) {

	__shared__ double cache[threadsPerBlock];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	double tmp;
	while (tid < n) {
		tmp += array[tid];
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = tmp;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) {
		result[blockIdx.x] = cache[0];
	}


}

double sum_vector_elements(double* array, int n) {
	double sum = 0;
	for (int i = 0; i < n; i++) {
		sum += array[i];
	}
	return sum;
}
void cpu_vector_sum(double *array, int n) {
	double time = clock();
	double sum = sum_vector_elements(array, n);
	cout << "Сумма элементов массива на CPU: " << sum << endl;
	cout << "Время работы программы на CPU (мс): " << clock() - time << endl;
}

void gpu_vector_sum(double* array, int n) {
	size_t byte_size = n * sizeof(double);

	double *h_array, *h_result;
	float time;

	int blocksPerGrid = 32 > ((n + threadsPerBlock - 1) / threadsPerBlock) ? 
		((n + threadsPerBlock - 1) / threadsPerBlock) : 32;

	double* result = new double[blocksPerGrid * sizeof(double)];

	cudaMalloc((void**)&h_array, byte_size);
	cudaMalloc((void**)&h_result, blocksPerGrid * sizeof(double));

	cudaMemcpy(h_array, array, byte_size, cudaMemcpyHostToDevice);

	vector_sum_kernel<<<blocksPerGrid, threadsPerBlock >>> (h_array, n, h_result);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMemcpy(result, h_result, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cout << "Сумма элементов массива на GPU: " << sum_vector_elements(result, blocksPerGrid) << endl;


	cout << "Время работы на GPU (мс): " << time << endl;

	cudaFree(result);
	cudaFree(h_array);
}


int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "Russian");

	srand(time(nullptr));

	cout << "Введите количество элементов массива: ";
	int N;
	cin >> N;

	double* array = new double[N];
	init_array(array, N);

	if (N < 10) {
		print_array(array, N);
	}

	cpu_vector_sum(array, N);


	gpu_vector_sum(array, N);

	return 0;
}
