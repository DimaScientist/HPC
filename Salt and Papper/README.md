# Лабораторная работа №3:Salt and pepper
***

## Постановка задачи:

Реализовать алгоритм медианного фильтра.
 
Язык: __Python__

Технологии: __PyCUDA__

Входные данные: изображение в оттенках серого.

Выходные данные: очищенное изображение + время вычисления.

## Описание работы программы на CUDA:

Идея реализации алгоритма на CUDA заключается в том, чтобы каждая нить в отдельности вычисляла медианное значение для каждого пикселя в окне.

Также вычисляем координаты нити:

```
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

Для вычисления медианного значения яркости для пикселя, сначала просмотрим яркости всех пикселей, которые находятся внутри окна. Обязательно производится проверка на нахождения нити внутри изображения.
Запишем все эти значения в массив и отсортируем его. Значение яркости пикселя будет находится в элементе индекса по середине, в нашем случае с индексом 4.

```
if ((x < height) && (y < width)) {
        int mask[COUNT_POINTS];
        int counter = 0;
        
        for (int j = - 1; j <= + 1; j++) {
            for (int i = - 1; i <= + 1; i++) {
                mask[counter] = tex2D(texture_ref, y + i, x + j);
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

        image_out[x * width + y] = mask[4];

    }

```

В ходе создания программы на CUDA была использована текстурная память ```texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> texture_ref```, которая привязывалась к исходному изображению.

Подробнее посмотреть на примеры работы и реализацию можно в файле [Median filter.ipynb](https://github.com/DimaScientist/HPC/blob/main/Salt%20and%20Papper/Median%20filter.ipynb).


## Пример работы программы:

Исходное изображение, которое было получено при запуске алгоритма наложения шума, который находится в [Salt and Papper noise.ipynb](https://github.com/DimaScientist/HPC/blob/main/Salt%20and%20Papper/Salt%20and%20Papper%20noise.ipynb):

![Исходное изображение](https://github.com/DimaScientist/HPC/blob/main/Salt%20and%20Papper/data/window_600x400.jpg)

Результат на __CPU__:

![Результат CPU](https://github.com/DimaScientist/HPC/blob/main/Salt%20and%20Papper/images/result_cpu.png)

Результат на __GPU__:

![Результат GPU](https://github.com/DimaScientist/HPC/blob/main/Salt%20and%20Papper/images/result_gpu.png)


## Результаты экспериментов:

Обозначения:

* n - размер картинки;

* t_gpu - время работы программы на CUDA;

* t_cpu - время работы программы на CPU;

* S = t_cpu / t_gpu - ускорение.

| параметр \ n | 600x400    | 1536x1024   | 1920x1280   | 2880x1500    | 5616x3744   |
| ------------ | ---------- | ----------- | ----------- | ------------ | ----------- | 
| t_cpu, с     |  1,402     | 9,288       | 14,61       | 25,513       | 126,082     | 
| t_gpu, с     | 0,002      | 0,007       | 0,009       | 0,02         | 0,086       | 
| S            | 701,1      | 1328,949    | 1623,268    | 1275,652     | 1465,683    | 

График зависимости времени работы программы на __CPU__ от размера каринки:

![График зависимости времени работы программы на CPU от размера картинки](https://github.com/DimaScientist/HPC/blob/main/Salt%20and%20Papper/images/cpu.png)

График зависимости времени работы программы на __GPU__ от размера картинки:

![График зависимости времени работы программы на GPU от размера картинки](https://github.com/DimaScientist/HPC/blob/main/Salt%20and%20Papper/images/gpu.png)

График зависимости __ускорения__ (boost) от размера картинки:

![График зависимости ускорения от размера картинки](https://github.com/DimaScientist/HPC/blob/main/Salt%20and%20Papper/images/boost.png)

## Выводы:

1. Видно явное превосходство программы с использованием CUDA (GPU) посравнению с CPU по времени выполненя;
2. Из графиков времени видно, что с увеличением размера картинки увеличивается и время работы программы. Однако из графика ускорения видно, что время работы программы на CPU растет быстрее, чем на GPU.
