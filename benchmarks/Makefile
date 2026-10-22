CC = aarch64-linux-gnu-gcc
CFLAGS = -O2 -static -Wall

TARGETS = matmul-arm stream-arm quicksort-arm hash-arm

all: $(TARGETS)

matmul-arm: matmul.c
	$(CC) $(CFLAGS) -o $@ $<

stream-arm: stream.c
	$(CC) $(CFLAGS) -o $@ $<

quicksort-arm: quicksort.c
	$(CC) $(CFLAGS) -o $@ $<

hash-arm: hash.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
