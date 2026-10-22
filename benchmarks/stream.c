#include <stdio.h>
#include <stdlib.h>

#define STREAM_ARRAY_SIZE 32768
#define NTIMES 4

static double a[STREAM_ARRAY_SIZE];
static double b[STREAM_ARRAY_SIZE];
static double c[STREAM_ARRAY_SIZE];

int main() {
    double scalar = 3.0;

    for (int i = 0; i < STREAM_ARRAY_SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    // Stream Triad: a[i] = b[i] + scalar * c[i] (Memory Bandwidth & DRAM Bound)
    for (int k = 0; k < NTIMES; k++) {
        for (int i = 0; i < STREAM_ARRAY_SIZE; i++) {
            a[i] = b[i] + scalar * c[i];
            c[i] = a[i] * 0.5;
        }
    }

    double checksum = 0.0;
    for (int i = 0; i < STREAM_ARRAY_SIZE; i += 64) {
        checksum += a[i] + c[i];
    }

    printf("stream completed. Checksum: %f\n", checksum);
    return 0;
}
