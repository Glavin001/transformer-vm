/* Count down from '9' to '0' and print each digit.
 * Tests: iadd, isub, icmp (sge), brif loop, putchar.
 * Single function, no helper calls, no string parsing. */

void compute(const char *input) {
    int n = 9;
    while (n >= 0) {
        putchar('0' + n);
        n = n - 1;
    }
    putchar('\n');
}
