/* pingo-ptbr edge in the browser: the same 4-bit acoustic model + int8 xxs vocoder in C, compiled to WebAssembly (no JS glue).
 * JS fills pingo_tok/pingo_stress/pingo_mask (L entries), calls pingo_speak and reads pingo_pcm (float, 16 kHz, with the 0.85 de-emphasis applied). */
#include <math.h>
#include <string.h>
#include "acoustic.h"
#include "fargan_xs.h"
#define EXPORT __attribute__((visibility("default")))
#define PINGO_MAXFRAMES 3000
static int tok[AC_MAXTOK], stress[AC_MAXTOK], mask[AC_MAXTOK];
static float pcm[PINGO_MAXFRAMES * XS_FRAME];
static float wsbuf[AC_MAXTOK * (2 * AC_W + 2 * AC_KVW)], enc[AC_MAXTOK * AC_W], ld[AC_MAXTOK];
static int dur[AC_MAXTOK];
static ac_dec_state ds; static xs_state vs;
EXPORT int *pingo_tok(void) { return tok; }
EXPORT int *pingo_stress(void) { return stress; }
EXPORT int *pingo_mask(void) { return mask; }
EXPORT float *pingo_pcm(void) { return pcm; }
EXPORT int pingo_maxtok(void) { return AC_MAXTOK; }

/* returns the number of samples, or -1 (too many tokens) / -2 (longer than 30 s) */
EXPORT int pingo_speak(int L, int voice, float speed) {
    if (L < 1 || L > AC_MAXTOK) return -1;
    static ac_enc_ws ws; ac_enc_ws_bind(&ws, wsbuf, L); long macs = 0;
    if (ac_encode(&ws, tok, stress, voice, L, enc, ld, &macs)) return -1;
    int T = ac_durations(ld, tok, mask, L, speed, dur); if (T < 1 || T > PINGO_MAXFRAMES) return -2;
    ac_dec_init(&ds); xs_init(&vs);
    float hist[5 * XS_NF], fr0[XS_NF], ex[XS_FRAME], y = 0.f; int t = 0, n = 0;
    #define EMIT() do { xs_frame(&vs, hist, ex); for (int i = 0; i < XS_FRAME; i++) { y = ex[i] + 0.85f * y; pcm[n * XS_FRAME + i] = y; } n++; } while (0)
    for (int i = 0; i < L; i++) for (int k = 0; k < dur[i]; k++, t++) {
        float fr = ((float)k + 0.5f) / (float)dur[i], ph[4] = { 2 * fr - 1, sinf(3.14159265f * fr), cosf(3.14159265f * fr), logf(1.f + dur[i]) / logf(1.f + AC_CFG_MAX_TOKEN_FRAMES) };
        ac_dec_frame(&ds, enc + i * AC_W, ph, t, fr0);
        if (t == 0) { for (int j = 0; j < 5; j++) memcpy(hist + j * XS_NF, fr0, sizeof fr0); }
        else { memmove(hist, hist + XS_NF, 4 * XS_NF * sizeof(float)); memcpy(hist + 4 * XS_NF, fr0, sizeof fr0); EMIT(); }
    }
    memmove(hist, hist + XS_NF, 4 * XS_NF * sizeof(float)); memcpy(hist + 4 * XS_NF, fr0, sizeof fr0); EMIT();
    return n * XS_FRAME;
}
