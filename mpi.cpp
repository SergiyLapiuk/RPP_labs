#include <iostream>
#include <cmath>
#include <mpi.h>

float f(float x) {
    return expf(-(x * x));
}

float Simpson(float a, float b, int n, float h) {
    float integral = f(a) + f(b);
    float buffer = 0.0;

    for (int i = 1; i <= n - 1; i += 2) {
        buffer += f(a + i * h);
    }
    integral += 4 * buffer;
    buffer = 0.0;

    for (int i = 2; i <= n - 2; i += 2) {
        buffer += f(a + i * h);
    }
    integral += 2 * buffer;

    integral *= h / 3;

    return integral;
}

void Get_data(float* a_ptr, float* b_ptr, int* n_ptr, int my_rank, int p, MPI_Comm comm) {
    if (my_rank == 0) {
        std::cout << "Enter a, b, and n\n";
        std::cin >> *a_ptr >> *b_ptr >> *n_ptr;
    }
    MPI_Bcast(a_ptr, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(b_ptr, 1, MPI_FLOAT, 0, comm);
    MPI_Bcast(n_ptr, 1, MPI_INT, 0, comm);
}


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

    // Розрахунок локальних меж на кореневому процесі
    float* local_as = nullptr;
    float* local_bs = nullptr;
    if (my_rank == 0) {
        local_as = new float[p];
        local_bs = new float[p];
        for (int i = 0; i < p; i++) {
            local_as[i] = a + i * n / p * h;
            local_bs[i] = local_as[i] + n / p * h;
        }
    }

    // Розповсюдження локальних меж
    MPI_Scatter(local_as, 1, MPI_FLOAT, &local_a, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(local_bs, 1, MPI_FLOAT, &local_b, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    integral = Simpson(local_a, local_b, n / p, h);

    elapsed_time += MPI_Wtime();

    MPI_Reduce(&integral, &total, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        std::cout << "With n = " << n << " subintervals, our estimate of the integral from " << a << " to " << b << " = " << total << "\n";
        std::cout << "Time taken " << elapsed_time << " seconds.\n";
        delete[] local_as;
        delete[] local_bs;
    }

    MPI_Finalize();
    return 0;
}
