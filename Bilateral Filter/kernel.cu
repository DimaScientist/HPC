
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <algorithm>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16

using namespace std;
using namespace cv;

float cpu_gauss[64];

__constant__ float gpu_gauss[64];

texture<unsigned char, 2, cudaReadModeElementType> texture_ref;

void update_gauss_gpu(int radius, double sigma) {

	float gauss_array[64];
	for (int i = 0; i < 2 * radius + 1; i++) {
		float x = i - radius;
		gauss_array[i] = expf(-(x * x) / (2 * sigma * sigma));
	}
	cudaMemcpyToSymbol(gpu_gauss, gauss_array, sizeof(float) * (2 * radius + 1));
}


void update_gauss_cpu(int radius, float sigma) {

	for (int i = 0; i < 2 * radius + 1; i++) {
		int x = i - radius;
		cpu_gauss[i] = exp(-(x * x) / (2 * sigma * sigma));
	}
}

__device__ inline double kernel_gaussian(float x, double sigma) {
	return __expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

float euclid_distance(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2)));
}

__global__ void kernel_bilateral_filter(unsigned char* input, unsigned char* output, int width, int height,
	int filter_radius, double sigma)
{
	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	if ((x < width) && (y < height)) {
		double t = 0;
		double sumFactor = 0;
		unsigned char center = tex2D(texture_ref, x, y);

		for (int i = -filter_radius; i <= filter_radius; i++) {
			for (int j = -filter_radius; j <= filter_radius; j++) {

				unsigned char curPix = tex2D(texture_ref, x + j, y + i);

				double factor = (gpu_gauss[i + filter_radius] * gpu_gauss[j + filter_radius]) * kernel_gaussian(center - curPix, sigma);

				t += factor * curPix;
				sumFactor += factor;
			}
		}
		output[y * width + x] = t / sumFactor;
	}
}

void bilateral_filter_gpu(const Mat& input, Mat& output, int filter_radius, double sigma) {
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int gray_size = input.step * input.rows;

	size_t pitch;
	unsigned char* d_input = NULL;
	unsigned char* d_output;

	update_gauss_gpu(filter_radius, sigma);

	// Allocate device memory
	cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char) * input.step, input.rows);
	cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char) * input.step, sizeof(unsigned char) * input.step, input.rows, cudaMemcpyHostToDevice);
	cudaBindTexture2D(0, texture_ref, d_input, input.step, input.rows, pitch);
	cudaMalloc<unsigned char>(&d_output, gray_size);

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);

	dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	cudaEventRecord(start, 0);

	kernel_bilateral_filter << <grid, block >> > (d_input, d_output, input.cols, input.rows, filter_radius, sigma);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(output.ptr(), d_output, gray_size, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventElapsedTime(&time, start, stop);
	cout << "Время GPU: " << time << " мc" << endl;
}

// CPU
void bilateral_filter_cpu(unsigned char* input, unsigned char* output, int width, int height, int filter_radius, float sigma) {
	update_gauss_cpu(filter_radius, sigma);

	float domainDist, colorDist, factor;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float t = 0.0f;
			float sum = 0.0f;

			for (int i = -filter_radius; i <= filter_radius; i++) {
				int neighborY = y + i;
				if (neighborY < 0) {
					neighborY = 0;
				}
				else if (neighborY >= height) {
					neighborY = height - 1;
				}
				for (int j = -filter_radius; j <= filter_radius; j++) {
					domainDist = cpu_gauss[filter_radius + i] * cpu_gauss[filter_radius + j];

					int neighborX = x + j;

					if (neighborX < 0) {
						neighborX = 0;
					}
					else if (neighborX >= width) {
						neighborX = width - 1;
					}
					colorDist = euclid_distance(input[neighborY * width + neighborX] - input[y * width + x], sigma);
					factor = domainDist * colorDist;
					sum += factor;
					t += factor * input[neighborY * width + neighborX];
				}
			}
			output[y * width + x] = t / sum;
		}
	}
}

int main(int argc, char** argv)
{
	setlocale(LC_ALL, "Russian");
	cout << "Введите путь до изображения: ";

	string path;
	cin >> path;


	cout << "Введите стандартное отклонение, $/sigma$: ";

	float sigma;
	cin >> sigma;

	Mat input_image = imread(path, IMREAD_GRAYSCALE);

	Mat output_gpu(input_image.rows, input_image.cols, CV_8UC1);
	Mat output_cpu(input_image.rows, input_image.cols, CV_8UC1);

	bilateral_filter_gpu(input_image, output_gpu, 7, sigma);

	clock_t start_s = clock();
	bilateral_filter_cpu(input_image.ptr(), output_cpu.ptr(), input_image.rows, input_image.cols, 7, sigma);
	clock_t stop_s = clock();
	cout << "Время CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " мс" << endl;
	cout << "Размеры изображения: " << input_image.rows << " x " << input_image.cols << endl;

	imshow("Результирующее изображение", output_gpu);
	waitKey(0);

	return 0;
}
