#include <stdio.h>
#include <stdint.h>

#define HASH_ITERS 100000

// Murmur-inspired 32-bit integer mixing (Integer ALU, bit shift, ILP bound)
static inline uint32_t murmur_mix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

int main() {
    uint32_t state = 0x12345678;
    uint32_t accum = 0;

    for (int i = 0; i < HASH_ITERS; i++) {
        state = murmur_mix32(state ^ (uint32_t)i);
        accum += state;
    }

    printf("hash completed. Final accum: %u\n", accum);
    return 0;
}
