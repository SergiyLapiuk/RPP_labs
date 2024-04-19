#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include "mpi.h"
#include <time.h>


// Кількість рядків і стовпців у матриці
#define N 2001

MPI_Status status;

// Створення матриць
double matrix_a[N][N], matrix_b[N][N], matrix_c[N][N];

int main(int argc, char** argv)
{
    int processCount, processId, slaveTaskCount, source, dest, rows, offset;

    // Ініціалізація середовища MPI 
    MPI_Init(&argc, &argv);
    // Кожен процес отримує унікальний ID (ранг)
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    // Кількість процесів у комунікаторі буде присвоєно змінній -> processCount
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    // Кількість підлеглих завдань буде призначено змінній -> slaveTaskCount
    slaveTaskCount = processCount - 1;

    // Кореневий (головний) процес
    if (processId == 0) {

        // Матриці A і B будуть заповнені випадковими числами
        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix_a[i][j] = rand() % 10;
                matrix_b[i][j] = rand() % 10;
            }
        }

        double startTime = MPI_Wtime();

        printf("\n\t\tMatrix - Matrix Multiplication using MPI\n");

        //// Роздрукувати матрицю A
        //printf("\nMatrix A\n\n");
        //for (int i = 0; i < N; i++) {
        //    for (int j = 0; j < N; j++) {
        //        printf("%.0f\t", matrix_a[i][j]);
        //    }
        //    printf("\n");
        //}

        //// Друк матриці B
        //printf("\nMatrix B\n\n");
        //for (int i = 0; i < N; i++) {
        //    for (int j = 0; j < N; j++) {
        //        printf("%.0f\t", matrix_b[i][j]);
        //    }
        //    printf("\n");
        //}

        // Визначте кількість рядків матриці A, які надсилаються кожному підпорядкованому процесу
        rows = N / slaveTaskCount;
        // Змінна зміщення визначає початкову точку рядка, який надсилається підпорядкованому процесу
        offset = 0;

        // Деталі розрахунку призначаються підлеглим завданням. Процес 1 і далі;
        // Тег кожного повідомлення дорівнює 1
        for (dest = 1; dest <= slaveTaskCount; dest++)
        {
            // Визнаючи зміщення матриці A
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            // Підтвердження кількості рядів
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            // Надсилаємо рядки матриці A, які будуть призначені підпорядкованому процесу для обчислення
            MPI_Send(&matrix_a[offset][0], rows * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            // Матриця B надсилається
            MPI_Send(&matrix_b, N * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);

            // Зміщення змінюється відповідно до кількості рядків, надісланих кожному процесу
            offset = offset + rows;
        }

        // Кореневий процес чекає, поки кожен підлеглий процес надішле свій обчислений результат із тегом повідомлення 2
        for (int i = 1; i <= slaveTaskCount; i++)
        {
            source = i;
            // Отримати зсув певного підлеглого процесу
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            // Отримайте кількість рядків, які обробив кожен підпорядкований процес
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            // Обчислені рядки кожного процесу будуть збережені в матриці C відповідно до їх зсуву та
            // кількість оброблених рядків
            MPI_Recv(&matrix_c[offset][0], rows * N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }

        double endTime = MPI_Wtime();

        //// Print the result matrix
        //printf("\nResult Matrix C = Matrix A * Matrix B:\n\n");
        //for (int i = 0; i < N; i++) {
        //    for (int j = 0; j < N; j++)
        //        printf("%.0f\t", matrix_c[i][j]);
        //    printf("\n");
        //}
        printf("\n");

        printf("\n");

        // Виведення загального часу множення
        printf("Time taken for matrix multiplication: %f seconds\n", endTime - startTime);
    }

    // Підпорядкований процес
    if (processId > 0) {

        // Визначено ідентифікатор вихідного процесу
        source = 0;

        // Підлеглий процес очікує на буфери повідомлень із тегом 1, надіслані кореневим процесом
        // Кожен процес отримає та виконає це окремо на своїх процесах

        // Підлеглий процес отримує значення зміщення, надіслане кореневим процесом
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        // Підлеглий процес отримує кількість рядків, надісланих кореневим процесом
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        // Підлеглий процес отримує підчастину матриці A, призначену Root
        MPI_Recv(&matrix_a, rows * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        // Підпорядкований процес отримує матрицю B
        MPI_Recv(&matrix_b, N * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

        // Матричне множення

        for (int k = 0; k < N; k++) {
            for (int i = 0; i < rows; i++) {
                // Встановити початкове значення підсумовування рядка
                matrix_c[i][k] = 0.0;
                // Matrix A's element(i, j) will be multiplied with Matrix B's element(j, k)
                for (int j = 0; j < N; j++)
                    matrix_c[i][k] = matrix_c[i][k] + matrix_a[i][j] * matrix_b[j][k];
            }
        }

        // Обчислений результат буде надіслано назад до кореневого процесу (процес 0) з тегом повідомлення 2

        // Зміщення буде надіслано до кореня, який визначає початкову точку обчисленого
        // значення в матриці C
        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        // Кількість рядків, яку розрахував процес, буде надіслано до кореневого процесу
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        // Результуюча матриця з обчисленими рядками буде надіслана до кореневого процесу
        MPI_Send(&matrix_c, rows * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}