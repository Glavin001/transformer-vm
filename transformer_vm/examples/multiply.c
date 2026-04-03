/* Multiply input length by 7 and output the result digit by digit.
 * Tests: imul (integer multiply), iadd, icmp, output.
 * Single function, no helper calls. */

void compute(const char *input) {
    int len = 0;
    while (input[len]) len = len + 1;

    int result = len * 7;

    /* Output digits (result is at most 2 digits for reasonable input). */
    if (result >= 10) {
        putchar('0' + result / 10);
    }
    putchar('0' + result % 10);
    putchar('\n');
}
