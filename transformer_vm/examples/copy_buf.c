/* Write N bytes to a buffer, then read them back and output.
 * Tests: store8 + uload8 in a loop with computed addresses.
 * This is the minimal pattern that reverse.c uses.
 * Single function, no helper calls. */

void compute(const char *input) {
    /* Count input length. */
    int len = 0;
    while (input[len]) len = len + 1;

    /* Copy input to buffer at fixed offset (simpler than reversing). */
    int buf_off = 16;
    int i = 0;
    while (i < len) {
        ((char *)input)[buf_off + i] = input[i];
        i = i + 1;
    }

    /* Output from buffer. */
    i = 0;
    while (i < len) {
        putchar(input[buf_off + i]);
        i = i + 1;
    }
    putchar('\n');
}
