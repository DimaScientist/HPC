# Лабораторная работа №0: Перемножение матриц
***

## Постановка задачи:

Реализовать алгоритм перемножения матриц.
 
Язык: __C++__

Входные данные: 2 матрицы размером от 100х100 до 2000х2000 каждая.

Выходные данные: проверка корректности перемножения + время вычисления.

## Описание работы программы на CUDA:

Пусть даны матрицы *A[n x m]* и *B[m x l]*, тогда *AB = C[n x l]*.
Каждый элемент матрицы *C* вычисляется следующим образом: *с[i][j] += a[i][k] x b[k][j], i=1..n,j=1..l,k=1..m*.
Из этого следует, что логичней распараллелить процесс вычисления отдельного элемента новой результирующей матрицы между нитями.

Для каждого блока и нити находим координаты элемента результирующей матрицы:

```
int row = blockDim.y * blockIdx.y + threadIdx.y;
int col = blockDim.x * blockIdx.x + threadIdx.x;
```

Поскольку используются квадратные матрицы, то *n=m=l*. С помощью цикла, проходя по строке и столбцу, находим сумму произведений:

```
C[n * row + col] = 0;

    if (row < n && col < n) {
        for (int k = 0; k < n; k++) {
            C[n * row + col] += A[row * n + k] * B[k * n + col];
        }
    }
```

При этом количество блоков в сетке и нитей в блоке: 

```
dim3 threadsPerBlock(n, n);
dim3 blocksPerGrid(1, 1);
```

Но максимальное количество нитей на блок __1024__. Поэтому для больших размеров матриц, размер блока будет 16x16, и будем увеличивать сетку блоков следующим образом:

```
#define BLOCK_SIZE 16

...

threadsPerBlock.x = BLOCK_SIZE;
threadsPerBlock.y = BLOCK_SIZE;
blocksPerGrid.x = ceil((n + threadsPerBlock.x - 1) / threadsPerBlock.x);
blocksPerGrid.y = ceil((n + threadsPerBlock.y - 1) / threadsPerBlock.y);
```


## Пример работы программы:

Пример работы программы при размерах матриц 128x128:

![Работа программы для матриц размера 128x128](https://github.com/DimaScientist/HPC/blob/main/MatMul/images/work.jpg)


## Результаты экспериментов:

Обозначения:

* n - размер матрицы;

* t_gpu - время работы программы на CUDA;

* t_cpu - время работы программы на CPU;

* S = t_cpu / t_gpu - ускорение.

| параметр \ n | 100    | 128   | 256   | 512    | 1024   | 2048      |
| ------------ | ------ | ----- | ----- | ------ | ------ | --------- |
| t_cpu, мс    |  3     | 6     | 48    | 387    | 3274   | 173483    |
| t_gpu, мс    | 0,107  | 0,143 | 0,8   | 7,29   | 57,286 | 405,859   |
| S            | 28,037 | 42    | 60    | 53,086 | 57,152 | 427,446   |

График зависимости времени работы программы на __CPU__ от размера матрицы:

![График зависимости времени работы программы на CPU от размера матрицы](https://github.com/DimaScientist/HPC/blob/main/MatMul/images/cpu.png)

График зависимости времени работы программы на __GPU__ от размера матрицы:

![График зависимости времени работы программы на GPU от размера матрицы](https://github.com/DimaScientist/HPC/blob/main/MatMul/images/gpu.png)

График зависимости __ускорения__ от размера матрицы:

![График зависимости ускорения от размера матрицы](https://github.com/DimaScientist/HPC/blob/main/MatMul/images/boost.png)

## Выводы:

1. Программа с использованием CUDA (GPU) работает бысрее, чем на CPU;
2. С увеличением размера матрицы увеличивается и время работы программы. По графику ускорения видно, что время работы программы на CPU растёт быстрее, чем на GPU.
