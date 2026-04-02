/* Output 'A', 'B', 'C' using a simple while(i<3) loop.
 * Tests: the basic while(i<N) loop pattern with hardcoded bound. */

void compute(const char *input) {
    int i = 0;
    while (i < 3) {
        putchar('A' + i);
        i = i + 1;
    }
    putchar('\n');
}
