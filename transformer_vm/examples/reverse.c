/* Reverse the input string and print it.
 * Tests: store8 + uload8 in combination, computed addresses.
 * Single function, no helper calls.
 * Uses explicit pointer decrement to avoid Cranelift bxor/bnot optimizations. */

void compute(const char *input) {
    /* Count input length. */
    int len = 0;
    while (input[len]) len = len + 1;

    /* Write reversed characters starting at a fixed offset after input. */
    int buf_off = 16;  /* Fixed offset, avoids len+1 edge case */
    int i = 0;
    int j = len - 1;
    while (i < len) {
        ((char *)input)[buf_off + i] = input[j];
        i = i + 1;
        j = j - 1;
    }

    /* Output the reversed string. */
    i = 0;
    while (i < len) {
        putchar(input[buf_off + i]);
        i = i + 1;
    }
    putchar('\n');
}
