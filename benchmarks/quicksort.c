#include <stdio.h>
#include <stdlib.h>

#define QS_SIZE 16384

static int arr[QS_SIZE];

static unsigned long next_rand = 1;
static int simple_rand(void) {
    next_rand = next_rand * 1103515245 + 12345;
    return (unsigned int)(next_rand / 65536) % 32768;
}

static void quicksort(int *a, int left, int right) {
    if (left >= right) return;
    int pivot = a[(left + right) / 2];
    int i = left;
    int j = right;
    
    // Irregular control-flow and branch mispredictions
    while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) {
            int tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
            i++;
            j--;
        }
    }
    quicksort(a, left, j);
    quicksort(a, i, right);
}

int main() {
    for (int i = 0; i < QS_SIZE; i++) {
        arr[i] = simple_rand();
    }

    quicksort(arr, 0, QS_SIZE - 1);

    long long sum = 0;
    for (int i = 0; i < QS_SIZE; i += 32) {
        sum += arr[i];
    }

    printf("quicksort completed. Sample sum: %lld\n", sum);
    return 0;
}
