/* Reverse the input string and print it.
 * Tests: store8 + uload8 in combination, computed addresses, two-pass loop.
 * Single function, no helper calls. */

void compute(const char *input) {
    int len = 0;
    while (input[len]) len = len + 1;

    /* Write reversed characters to buffer after the input's null terminator. */
    char *buf = (char *)(input + len + 1);
    int i = 0;
    int j = len - 1;
    while (i < len) {
        buf[i] = input[j];
        i = i + 1;
        j = j - 1;
    }

    /* Output the reversed string. */
    i = 0;
    while (i < len) {
        putchar(buf[i]);
        i = i + 1;
    }
    putchar('\n');
}
