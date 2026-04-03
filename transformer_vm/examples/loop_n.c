/* Output N copies of 'X' where N is the input length.
 * Tests: while(i < len) loop where len is computed at runtime. */

void compute(const char *input) {
    int len = 0;
    while (input[len]) len = len + 1;

    int i = 0;
    while (i < len) {
        putchar('X');
        i = i + 1;
    }
    putchar('\n');
}
