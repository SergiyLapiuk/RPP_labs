#include <iostream>
#include <cmath>
#include <mpi.h>

float Simpson(float a, float b, int n, float h) {
    float integral;
    float x;
    float buffer;  // Тимчасове зберігання розрахунків
    int i;

    float f(float x);  // Функція, яку ми будемо інтегрувати

    integral = f(a) + f(b);  // Ініціалізуємо інтеграл за допомогою f(a) + f(b)
    buffer = 0.0;  // Ініціалізація буферу

    for (i = 1; i <= n - 1; i += 2) { // Примітка: n завжди має бути парним
        x = a + i * h;
        buffer += f(x);
    }
    integral += 4 * buffer; // Множимо всі непарні позиції на 4
    buffer = 0.0;  // Скинути буфер

    for (i = 2; i <= n - 2; i += 2) {
        x = a + i * h;
        buffer += f(x);
    }
    integral += 2 * buffer; // Помножте всі парні позиції на 2

    integral *= h / 3;

    return integral;
} // Кінець Сімпсона

// Функція Get_data: читає введені користувачем a, b і n
void Get_data(float* a_ptr, float* b_ptr, int* n_ptr, int my_rank, int p, MPI_Comm comm) {
    if (my_rank == 0) {
        std::cout << "Enter a, b, and n\n";
        std::cin >> *a_ptr >> *b_ptr >> *n_ptr;
    }

    // Розповсюдження a, b і n за допомогою BROADCAST
    MPI_Bcast(a_ptr, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(b_ptr, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(n_ptr, 1, MPI_INT, 0, comm);
} // Кінець Get_data

float f(float x) {   // Функція для оцінки e^(-x^2)
    return expf(-(x * x));
} // Кінець f

int main(int argc, char** argv) {
    int my_rank, p;
    float a, b;
    int n;
    float h;
    float local_a, local_b;
    int local_n;
    float integral, total;
    double elapsed_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    Get_data(&a, &b, &n, my_rank, p, MPI_COMM_WORLD);

    h = (b - a) / n;
    local_n = n / p;

    local_a = a + my_rank * local_n * h;
    local_b = local_a + local_n * h;

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    integral = Simpson(local_a, local_b, local_n, h);

    elapsed_time += MPI_Wtime();
    
    MPI_Reduce(&integral, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        std::cout << "With n = " << n << " subintervals, our estimate of the integral from " << a << " to " << b << " = " << total << "\n";
        std::cout << "Time taken " << elapsed_time << " seconds.\n";
    }

    MPI_Finalize();
    return 0;
} // Завершення  main
