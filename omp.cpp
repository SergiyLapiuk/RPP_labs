#include <iostream>
#include <cmath>
#include <omp.h>

float f(float x) {   // Функція для оцінки e^(-x^2)
    return expf(-(x * x));
} // Кінець f

float Simpson(float a, float b, int n, float h) {
    float integral = f(a) + f(b);
    float buffer_odd = 0.0;
    float buffer_even = 0.0;
    int i;

    // Обрахунок для непарних індексів
#pragma omp parallel for reduction(+:buffer_odd) default(none) shared(a, h, n) private(i)
    for (i = 1; i <= n - 1; i += 2) {
        float x = a + i * h;
        buffer_odd += f(x);
    }

    // Обрахунок для парних індексів
#pragma omp parallel for reduction(+:buffer_even) default(none) shared(a, h, n) private(i)
    for (i = 2; i <= n - 2; i += 2) {
        float x = a + i * h;
        buffer_even += f(x);
    }

    integral += 4 * buffer_odd + 2 * buffer_even;
    integral *= h / 3;

    return integral;
} // Кінець Сімпсона

void Get_data(float* a_ptr, float* b_ptr, int* n_ptr) {
    std::cout << "Enter a, b, and n\n";
    std::cin >> *a_ptr >> *b_ptr >> *n_ptr;
} // Кінець Get_data

int main() {
    float a, b;
    int n;
    float h, integral;

    Get_data(&a, &b, &n);

    h = (b - a) / n;

    // Задайте кількість потоків
    int num_threads;
    std::cout << "Enter the number of threads: ";
    std::cin >> num_threads;
    omp_set_num_threads(num_threads);

    double start_time = omp_get_wtime();
    integral = Simpson(a, b, n, h);
    double elapsed_time = omp_get_wtime() - start_time;

    std::cout << "With n = " << n << " subintervals, our estimate of the integral from " << a << " to " << b << " = " << integral << "\n";
    std::cout << "Time taken " << elapsed_time << " seconds.\n";

    return 0;
} // Завершення main
