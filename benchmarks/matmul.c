#include <stdio.h>
#include <stdlib.h>

#define N 64

// Statically allocated matrices to avoid dynamic heap overhead in gem5 SE mode
static double A[N][N];
static double B[N][N];
static double C[N][N];

int main() {
    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)(i + j) * 0.01;
            B[i][j] = (double)(i - j) * 0.02;
            C[i][j] = 0.0;
        }
    }

    // Dense Matrix Multiplication (Compute & Cache Bound)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = A[i][k];
            for (int j = 0; j < N; j++) {
                C[i][j] += r * B[k][j];
            }
        }
    }

    // Prevent compiler dead-code elimination
    double checksum = 0.0;
    for (int i = 0; i < N; i++) {
        checksum += C[i][i];
    }

    printf("matmul completed. Checksum: %f\n", checksum);
    return 0;
}
