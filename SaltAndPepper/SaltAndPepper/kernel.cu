
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>


using namespace std;
using namespace cv;

const int WINDOW_SIZE = 3;
const int COUNT_POINTS = 9;
const int BLOCK_SIZE = 16;


texture<float, cudaTextureType2D, cudaReadModeElementType> texture_ref;


__global__ void median_filter_kernel(unsigned char* image_in, unsigned char* image_out, int height, int width) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float mask[COUNT_POINTS] = {0, 0, 0, 0, 0, 0, 0, 0};


    if (x < width && y < height) {
        int counter = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                mask[counter] = tex2D(texture_ref, x + j, y + i);
                counter++;
            }
        }


        for (int i = 1; i < COUNT_POINTS; i++) {
            for (int j = i; j > 0 && mask[j - 1] > mask[j]; j--) {
                int tmp = mask[j - 1];
                mask[j - 1] = mask[j];
                mask[j] = tmp;
            }
        }

        image_out[y * width + x] = mask[4];

    }
}



void median_filter_gpu(const Mat& input, Mat& output) {

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int image_size = input.cols * input.rows;

    size_t pitch;

    unsigned char* d_input = NULL;
    unsigned char* d_output;

    cudaMallocPitch(&d_input, &pitch, sizeof(unsigned char) * input.step, input.rows);
    cudaMemcpy2D(d_input, pitch, input.ptr(), sizeof(unsigned char) * input.step, sizeof(unsigned char) * input.step, input.rows, cudaMemcpyHostToDevice);

    cudaBindTexture2D(NULL, texture_ref, d_input, input.step, input.rows, pitch);
    cudaMalloc<unsigned char>(&d_output, image_size);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((input.cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (input.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start, 0);


    median_filter_kernel << <blocksPerGrid, threadsPerBlock >> > (d_input, d_output, input.rows, input.cols);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaMemcpy(output.ptr(), d_output, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventElapsedTime(&time, start, stop);
    cout << "Время GPU: " << time << " мc" << endl;
}

void median_filter_cpu(unsigned char* input, unsigned char* output, int height, int width) {

    float mask[COUNT_POINTS];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
                output[y * width + x] = 0;
            }
            else {
                for (int i = 0; i < WINDOW_SIZE; i++) {
                    for (int j = 0; j < WINDOW_SIZE; j++) {
                        mask[i * WINDOW_SIZE + j] = input[(y + i - 1)* width + (j + x - 1)];
                    }
                }

                for (int i = 1; i < COUNT_POINTS; i++) {
                    for (int j = i; j > 0 && mask[j - 1] > mask[j]; j--) {
                        int tmp = mask[j - 1];
                        mask[j - 1] = mask[j];
                        mask[j] = tmp;
                    }
                }

                output[y * width + x] = mask[4];
            }

        }
    }

}

int main()
{
    setlocale(LC_ALL, "RUS");
    cout << "Введите путь до изображения: ";

    string path;
    cin >> path;

    Mat input_image = imread(path, IMREAD_GRAYSCALE);

    cout << "Размеры входного изображения: " << input_image.rows << " x " << input_image.cols << endl;
    cout << input_image.step << endl;

    Mat output_gpu(input_image.rows, input_image.cols, CV_8UC1);
    Mat output_cpu(input_image.rows, input_image.cols, CV_8UC1);

    clock_t time_start = clock();
    median_filter_cpu(input_image.ptr(), output_cpu.ptr(), input_image.rows, input_image.cols);
    clock_t time_end = clock();
    cout << "Время CPU: " << (time_end - time_start) / double(CLOCKS_PER_SEC) * 1000 << " мс" << endl;

    imshow("Результат CPU", output_cpu);
    waitKey(30);

    median_filter_gpu(input_image, output_gpu);

    imshow("Результат GPU", output_gpu);
    waitKey(0);
   
    return 0;
}