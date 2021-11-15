
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


// маски Гауссовского размытия
float cpu_gauss[64];
__constant__ float gpu_gauss[64];

// текстурная память
texture<unsigned char, 2, cudaReadModeElementType> texture_ref; 

void update_gauss_gpu(int radius, double sigma) {

	/*
	Размытие по Гауссу для GPU
	*/

	float gauss_array[64];
	for (int i = 0; i < 2 * radius + 1; i++) {

		// обновление амплитуды Гауссова
		float x = i - radius;
		gauss_array[i] = expf(-(x * x) / (2 * sigma * sigma));
	}
	cudaMemcpyToSymbol(gpu_gauss, gauss_array, sizeof(float) * (2 * radius + 1));
}


void update_gauss_cpu(int radius, float sigma) {

	/*
	Размытие по Гауссу для CPU
	*/

	for (int i = 0; i < 2 * radius + 1; i++) {
		// обновление амплитуды Гауссова

		int x = i - radius;
		cpu_gauss[i] = exp(-(x * x) / (2 * sigma * sigma));
	}
}

__device__ inline double kernel_euclid_distance(float x, double sigma) {

	/*
	Евклидово расстояние на GPU
	*/

	return __expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

float euclid_distance(float x, double sigma) {

	/*
	Евклидово расстояние на CPU
	*/

	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2)));
}

__global__ void kernel_bilateral_filter(unsigned char* input, unsigned char* output, int width, int height, int filter_radius, double sigma) {

	/*
	Kernel фильтра
	*/

	// вычисляются "координаты" нити
	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	// проверка на то, что нить обрабатывает пиксель внутри картинки
	if ((x < width) && (y < height)) {

		double h = 0;
		double k = 0;
		unsigned char center = tex2D(texture_ref, x, y); // сохраним координаты пикселя, относительно которого будет производится размытие

		// для всех пискселей, находящихся внутри радиуса, где содержится маска
		for (int i = -filter_radius; i <= filter_radius; i++) {
			for (int j = -filter_radius; j <= filter_radius; j++) {

				unsigned char curr_pixel = tex2D(texture_ref, x + j, y + i); // выберем пиксель внутри радиуса

				double factor = gpu_gauss[i + filter_radius] * gpu_gauss[j + filter_radius] * kernel_euclid_distance(center - curr_pixel, sigma); // вычисляем коэффициент Гауссова размытия

				// вычисляем новую яркость и записываем коэффициент
				h += factor * curr_pixel;
				k += factor;
			}
		}

		// проводим нормализацию
		output[y * width + x] = h / k;
	}
}

void bilateral_filter_gpu(const Mat& input, Mat& output, int filter_radius, double sigma) {
	/*
	Реализация фильтра на GPU
	*/

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int image_size = input.cols * input.rows; // сохраним размер рисунка

	size_t pitch;

	// выделим переменные для входного и выходного изображения для оработки на GPU
	unsigned char* d_input = NULL; 
	unsigned char* d_output;

	// инициируем маску Гауссова размытия
	update_gauss_gpu(filter_radius, sigma);

	// выделим место для двумерного массива изображения
	cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char) * input.step, input.rows);

	// запишем в буфер значения значения яркостей на картинке, привязав указатель
	cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char) * input.step, sizeof(unsigned char) * input.step, input.rows, cudaMemcpyHostToDevice);

	// привяжем ссылки на текстуры к их буферам
	cudaBindTexture2D(NULL, texture_ref, d_input, input.step, input.rows, pitch);
	cudaMalloc<unsigned char>(&d_output, image_size);

	// зададим количество нитей и блоков
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((input.cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (input.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	cudaEventRecord(start, 0);

	kernel_bilateral_filter <<< blocksPerGrid, threadsPerBlock >>> (d_input, d_output, input.cols, input.rows, filter_radius, sigma);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// передадим указателю на новое изображение ссылку на буфер результирующего
	cudaMemcpy(output.ptr(), d_output, image_size, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

	cudaEventElapsedTime(&time, start, stop);
	cout << "Время GPU: " << time << " мc" << endl;
}

void bilateral_filter_cpu(unsigned char* input, unsigned char* output, int width, int height, int filter_radius, float sigma) {

	/*
	Реализация фильтра на CPU
	*/

	update_gauss_cpu(filter_radius, sigma);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float h = 0.0f;
			float k = 0.0f;

			for (int i = -filter_radius; i <= filter_radius; i++) {
				int neighbor_y = y + i;
				if (neighbor_y < 0) {
					neighbor_y = 0;
				}
				else if (neighbor_y >= height) {
					neighbor_y = height - 1;
				}
				for (int j = -filter_radius; j <= filter_radius; j++) {

					int neighbor_x = x + j;

					if (neighbor_x < 0) {
						neighbor_x = 0;
					}
					else if (neighbor_x >= width) {
						neighbor_x = width - 1;
					}
					float factor = cpu_gauss[filter_radius + i] * cpu_gauss[filter_radius + j] * euclid_distance(input[neighbor_y * width + neighbor_x] - input[y * width + x], sigma);
					k += factor;
					h += factor * input[neighbor_y * width + neighbor_x];
				}
			}
			output[y * width + x] = h / k;
		}
	}
}

int main(int argc, char** argv)
{
	setlocale(LC_ALL, "Russian");
	cout << "Введите путь до изображения: ";

	string path;
	cin >> path;

	cout << "Введите стандартное отклонение: ";

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

	/*
	imshow("Результирующее изображение CPU", output_cpu);
	waitKey(30);
	*/

	imshow("Результирующее изображение GPU", output_gpu);
	waitKey(0);

	return 0;
}
