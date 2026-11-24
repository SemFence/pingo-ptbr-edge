/* pingo in plain C: phonemes -> acoustic model -> xxs vocoder -> 16 kHz s16 PCM.   ./pingo_c ac_test_in.i32 out.s16 [speed] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "acoustic.h"
#include "fargan_xs.h"
#define MAXFR 700
static void *load(const char *p, long *n) { FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); } fseek(f, 0, SEEK_END); long b = ftell(f); fseek(f, 0, SEEK_SET);
    void *x = malloc(b); if (fread(x, 1, b, f) != (size_t)b) exit(1); fclose(f); *n = b; return x; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
    long nb; int *in = load(argv[1], &nb); float speed = argc > 3 ? (float)atof(argv[3]) : 1.f;
    int Ln = in[0], voice = in[1]; int *tok = in + 2, *st = in + 2 + Ln, *mask = in + 2 + 2 * Ln;
    static ac_enc_ws ws; static float wsbuf[AC_MAXTOK * (2 * AC_W + 2 * AC_KVW)]; static float enc[AC_MAXTOK * AC_W], ld[AC_MAXTOK], feats[(MAXFR + 4) * XS_NF]; int dur[AC_MAXTOK]; long macs = 0;
    double t0 = now();
    ac_enc_ws_bind(&ws, wsbuf, Ln); if (ac_encode(&ws, tok, st, voice, Ln, enc, ld, &macs)) { fprintf(stderr, "frase longa\n"); return 1; }
    int T = ac_durations(ld, tok, mask, Ln, speed, dur); if (T > MAXFR) { fprintf(stderr, "frase longa (%d quadros)\n", T); return 1; }
    static ac_dec_state ds; ac_dec_init(&ds); float *f = feats + 3 * XS_NF; int t = 0;       /* 3 frames of padding at the front */
    for (int i = 0; i < Ln; i++) for (int k = 0; k < dur[i]; k++, t++) {
        float fr = ((float)k + 0.5f) / (float)dur[i], ph[4] = { 2 * fr - 1, sinf(3.14159265f * fr), cosf(3.14159265f * fr), logf(1.f + dur[i]) / logf(1.f + AC_CFG_MAX_TOKEN_FRAMES) };
        ac_dec_frame(&ds, enc + i * AC_W, ph, t, f + t * XS_NF);
    }
    double t1 = now();
    for (int j = 0; j < 3; j++) memcpy(feats + j * XS_NF, f, XS_NF * sizeof(float));
    memcpy(f + T * XS_NF, f + (T - 1) * XS_NF, XS_NF * sizeof(float));
    static xs_state vs; xs_init(&vs); FILE *o = fopen(argv[2], "wb"); float y = 0.f, ex[XS_FRAME];
    for (int n = 0; n < T; n++) {
        xs_frame(&vs, feats + (long)n * XS_NF, ex);
        for (int i = 0; i < XS_FRAME; i++) { y = ex[i] + 0.85f * y; float v = y * 32768.f; v = v > 32767.f ? 32767.f : (v < -32768.f ? -32768.f : v); int16_t s = (int16_t)lrintf(v); fwrite(&s, 2, 1, o); }
    }
    fclose(o); double t2 = now(), secs = T / 100.0;
    printf("%d tokens -> %d quadros = %.2f s de áudio\n", Ln, T, secs);
    printf("MAC por segundo de áudio: acústico %.1f M/s (decoder) + %.2f M (encoder, 1x) | vocoder %.1f M/s\n", ds.macs / secs / 1e6, macs / 1e6, vs.macs / secs / 1e6);
    printf("tempo no Mac, float escalar sem otimizar: acústico %.3f s + vocoder %.3f s = %.1fx tempo real (NÃO vale para o ESP32)\n", t1 - t0, t2 - t1, secs / (t2 - t0));
    printf("RAM de trabalho (float): encoder %zu + decoder %zu + features %zu + vocoder %zu bytes\n", sizeof ws, sizeof ds, sizeof feats, sizeof vs);
    return 0;
}
