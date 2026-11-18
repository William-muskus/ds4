/* Distributed layer slices exchange one hidden-state row per token, and the
 * graphs read and write the same rows.  Both sides size those buffers from
 * ds4_engine_hidden_f32_values().  GLM 5.3 carries four hyper-connection
 * streams; sizing its rows as one plain hidden row made pipeline execution
 * overrun the coordinator's buffers and feed the next stage garbage.
 * Model-free: uses the built-in shapes, no GGUF needed. */
#include <inttypes.h>
#include <stdio.h>

#include "ds4.h"

static int failures;

static void expect_values(const char *name, int shape_id, uint64_t want) {
    const uint64_t got = ds4_test_hidden_row_values(shape_id);
    if (got != want) {
        fprintf(stderr, "%s: hidden row values %" PRIu64 ", want %" PRIu64 "\n",
                name, got, want);
        failures++;
    }
}

int main(void) {
    expect_values("DeepSeek V4 Flash", 0, 4u * 4096u);
    expect_values("GLM 5.2", 1, 6144u);
    expect_values("GLM 5.3 Flash", 2, 4u * 4096u);
    if (failures) return 1;
    printf("hidden row values: PASS\n");
    return 0;
}
