/* Write two bytes to memory and read them back.
 * Tests: store8 + uload8 roundtrip with computed addresses.
 * Single function, no loops for the memory part. */

void compute(const char *input) {
    /* Use memory 10 bytes after input as a buffer. */
    char *buf = (char *)(input + 10);
    buf[0] = 'X';
    buf[1] = 'Y';
    putchar(buf[0]);
    putchar(buf[1]);
    putchar('\n');
}
