/* Find the maximum character in input using if/else (not smax).
 * Tests: copy_false (the else branch of a conditional update).
 * Single function, no helper calls. */

void compute(const char *input) {
    int best = 0;
    int i = 0;
    while (input[i]) {
        int c = input[i];
        /* Explicit if/else to force copy_false generation. */
        if (c > best) {
            best = c;
        }
        i = i + 1;
    }
    putchar(best);
    putchar('\n');
}
