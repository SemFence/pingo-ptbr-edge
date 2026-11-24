/* Confere o acústico em C contra o MLX e mede custo.  ./ac_test ac_test_in.i32 ac_test_ref.f32 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "acoustic.h"
static void *load(const char *p, long *n) { FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); } fseek(f, 0, SEEK_END); long b = ftell(f); fseek(f, 0, SEEK_SET);
    void *x = malloc(b); if (fread(x, 1, b, f) != (size_t)b) exit(1); fclose(f); *n = b; return x; }
int main(int argc, char **argv) {
    long nb, nr; int *in = load(argv[1], &nb); float *ref = load(argv[2], &nr);
    int Ln = in[0], voice = in[1]; int *tok = in + 2, *st = in + 2 + Ln, *mask = in + 2 + 2 * Ln;
    static ac_enc_ws ws; static float wsbuf[AC_MAXTOK * (2 * AC_W + 2 * AC_KVW)]; static float enc[AC_MAXTOK * AC_W], ld[AC_MAXTOK]; int dur[AC_MAXTOK]; long macs = 0;
    ac_enc_ws_bind(&ws, wsbuf, Ln); if (ac_encode(&ws, tok, st, voice, Ln, enc, ld, &macs)) { fprintf(stderr, "frase longa\n"); return 1; }
    double e = 0, r = 0; for (int i = 0; i < Ln; i++) { double d = ld[i] - ref[1 + i]; e += d * d; r += (double)ref[1 + i] * ref[1 + i]; }
    printf("log-duração: erro rms relativo %.4f%%  (%d tokens)\n", 100 * sqrt(e / r), Ln);
    int T = ac_durations(ld, tok, mask, Ln, 1.0f, dur), Tref = (int)ref[0];
    printf("quadros: C %d | MLX %d\n", T, Tref);
    static ac_dec_state ds; ac_dec_init(&ds); const float *rf = ref + 1 + Ln; double fe = 0, fr = 0; int t = 0; long m0 = macs;
    for (int i = 0; i < Ln; i++) for (int f = 0; f < dur[i]; f++, t++) {
        float frac = ((float)f + 0.5f) / (float)dur[i], ph[4] = { 2 * frac - 1, sinf(3.14159265f * frac), cosf(3.14159265f * frac), logf(1.f + dur[i]) / logf(1.f + AC_CFG_MAX_TOKEN_FRAMES) };
        float o[20]; ac_dec_frame(&ds, enc + i * AC_W, ph, t, o);
        if (t < Tref) for (int k = 0; k < 20; k++) { double d = o[k] - rf[t * 20 + k]; fe += d * d; fr += (double)rf[t * 20 + k] * rf[t * 20 + k]; }
    }
    printf("features: erro rms relativo %.4f%% sobre %d quadros\n", 100 * sqrt(fe / fr), t < Tref ? t : Tref);
    printf("MAC: encoder %.2f M (uma vez por frase) | decoder %.2f M para %d quadros = %.2f M/s de áudio\n", m0 / 1e6, ds.macs / 1e6, T, ds.macs / (T / 100.0) / 1e6);
    printf("RAM de trabalho (float): encoder %lu bytes | decoder %zu bytes | enc de saída %d bytes\n", ac_enc_ws_bytes(Ln), sizeof(ac_dec_state), Ln * AC_W * 4);
    return 0;
}
