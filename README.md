# Fourier Transform Cpp

Данный код был написан в рамках выполнения лабораторной работы по дисциплине компьютерное зрение. В проекте реализовано преобразование Фурье [прямым подходом](src/dft.cpp) и [методом Кули – Тьюки](src/fft.cpp) для одноканальных изображений opencv. В разделе `test` приведены примеры применения преобразования Фурье, в частности [свёртка](test/filter_test.cpp), [сравнение изображений](test/template_test.cpp) и [обратная свёртка](test/deconvolution_test.cpp).

## Запуск

``` bash
git clone https://github.com/KirillHit/fourier_transform_cpp.git
mkdir build
cd build
cmake ..
cmake --build .
make test
```