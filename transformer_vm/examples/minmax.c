/* Find the min and max ASCII values in the input string.
 * Tests: icmp (slt, sgt), conditional updates via select, uload8 loop.
 * Single function, no helper calls. */

void compute(const char *input) {
    int mn = 127;
    int mx = 0;
    int i = 0;
    while (input[i]) {
        int c = input[i];
        if (c < mn) mn = c;
        if (c > mx) mx = c;
        i = i + 1;
    }
    putchar(mn);
    putchar(mx);
    putchar('\n');
}
