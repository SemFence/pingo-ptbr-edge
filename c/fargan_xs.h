/* Reduced FARGAN vocoder (xs/xxs sizes) in plain C: per-row int8 weights, float activations. No malloc, no dependencies besides libm.
 * Usage: xs_init(&st); for each frame n: xs_frame(&st, feats + 20*n, out160)  (feats = frames n..n+4, 20 floats each).
 * For output frame n to line up with feats[n], the caller passes [3 copies of the first frame] + feats + [1 copy of the last]. */
#pragma once
#include <stdint.h>
#include "xs_weights.h"

#define XS_NF 20            /* floats por quadro de features */
#define XS_PE 12            /* pitch embedding dimension */
#define XS_SUB 40           /* amostras por subframe */
#define XS_NSUB 4
#define XS_FRAME (XS_SUB * XS_NSUB)
#define XS_FIN (2 * XS_SUB + XS_C + 4)   /* fwc0 input (one step) */

typedef struct {
    float exc_mem[256];
    float h1[XS_H1], h2[XS_H2], h3[XS_H3];
    float fwc_state[XS_FIN];
    int8_t fwc_q[XS_FIN];            /* only the int8-activation variant uses this state */
    float dense1[3][XS_D1];          /* ring with the fdense1 output of the last 3 frames */
    int primed;
    long macs;                       /* multiply-accumulate counter (measurement only) */
} xs_state;

void xs_init(xs_state *st);
/* feats: 5 consecutive frames (n .. n+4) x 20 floats. out: 160 excitation samples (before the 0.85 de-emphasis). */
void xs_frame(xs_state *st, const float *feats, float *out);
