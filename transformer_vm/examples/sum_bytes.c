/* Sum all input bytes and output as 32-bit value read back from memory.
 * Tests: load (32-bit word load), store (32-bit word store), iadd in loop.
 * Single function, no helper calls. */

void compute(const char *input) {
    /* Accumulate sum of ASCII values. */
    int sum = 0;
    int i = 0;
    while (input[i]) {
        sum = sum + input[i];
        i = i + 1;
    }

    /* Store sum as 32-bit word, then load it back and output byte by byte. */
    int *buf = (int *)(input + 32);
    *buf = sum;
    int loaded = *buf;

    /* Output each byte of the loaded value (little-endian). */
    putchar(loaded & 255);
    putchar((loaded >> 8) & 255);
    putchar('\n');
}
