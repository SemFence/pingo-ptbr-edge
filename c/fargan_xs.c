#include <math.h>
#include <string.h>
#include "fargan_xs.h"

static inline float clamp1(float x) { return x < -1.f ? -1.f : (x > 1.f ? 1.f : x); }
#ifdef XS_FAST_MATH   /* aproximação racional (a do Opus): erro máx 6e-5 em tanh, 3e-5 em sigmoid; ~10 ciclos em vez de centenas */
static inline float tanh_(float x) {
    float x2 = x * x, num = (0.60863042f * x2 + 96.39235687f) * x2 + 952.52801514f, den = (11.88600922f * x2 + 413.36801147f) * x2 + 952.72399902f;
    float r = num * x / den; return r < -1.f ? -1.f : (r > 1.f ? 1.f : r);
}
static inline float sigmoidf_(float x) { return 0.5f + 0.5f * tanh_(0.5f * x); }
#else
static inline float tanh_(float x) { return tanhf(x); }
static inline float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
#endif

/* y[i] = escala[i] * sum_j w[i][j] * x[j]   (w int8 por linha, x float) */
static void mv(xs_state *st, const int8_t *w, const float *s, int rows, int cols, const float *x, float *y) {
    for (int i = 0; i < rows; i++) {
        const int8_t *r = w + (long)i * cols; float acc = 0.f;
        for (int j = 0; j < cols; j++) acc += x[j] * (float)r[j];
        y[i] = acc * s[i];
    }
    st->macs += (long)rows * cols;
}
#define MV(st, name, x, y) mv(st, xs_##name##_w, xs_##name##_s, XS_##name##_ROWS, XS_##name##_COLS, x, y)

void xs_init(xs_state *st) { memset(st, 0, sizeof *st); }

static int period_of(float f18) {
    float p = f18 < -10.f ? -10.f : (f18 > 10.f ? 10.f : f18);
    int v = (int)rintf(256.f / exp2f(p + 1.5f));
    return v < 32 ? 32 : (v > 255 ? 255 : v);
}

static void glu(xs_state *st, const int8_t *w, const float *s, int n, float *x) {   /* x <- x * sigmoid(W x) */
    float g[256]; mv(st, w, s, n, n, x, g);
    for (int i = 0; i < n; i++) x[i] *= sigmoidf_(g[i]);
}
#define GLU(st, name, n, x) glu(st, xs_##name##_w, xs_##name##_s, n, x)

static void dense1_frame(xs_state *st, const float *f, int per, float *out) {
    float in[XS_NF + XS_PE];
    memcpy(in, f, XS_NF * sizeof(float));
    const int8_t *e = xs_pembed_w + (long)(per - 32) * XS_pembed_COLS;               /* embedding int8 por linha */
    for (int i = 0; i < XS_PE; i++) in[XS_NF + i] = (float)e[i] * xs_pembed_s[per - 32];
    MV(st, fdense1, in, out);
    for (int i = 0; i < XS_D1; i++) out[i] = tanh_(out[i]);
}

static void gru(xs_state *st, const int8_t *wi, const float *si, const int8_t *wh, const float *sh, int nin, int nh, const float *x, float *h) {
    float gi[3 * 64 * 3], gh[3 * 64 * 3];   /* até 3*H com H <= 192 */
    mv(st, wi, si, 3 * nh, nin, x, gi); mv(st, wh, sh, 3 * nh, nh, h, gh);
    for (int i = 0; i < nh; i++) {
        float r = sigmoidf_(gi[i] + gh[i]), z = sigmoidf_(gi[nh + i] + gh[nh + i]);
        float c = tanh_(gi[2 * nh + i] + r * gh[2 * nh + i]);
        h[i] = (1.f - z) * c + z * h[i];
    }
}

void xs_frame(xs_state *st, const float *feats, float *out) {
    int per[5]; for (int j = 0; j < 5; j++) per[j] = period_of(feats[j * XS_NF + 18]);
    /* condicionamento: dense1 dos quadros n+2, n+3, n+4 (anel), conv k=3, dense2 */
    if (!st->primed) { for (int j = 0; j < 2; j++) dense1_frame(st, feats + (2 + j) * XS_NF, per[2 + j], st->dense1[j + 1]); st->primed = 1; }
    memmove(st->dense1[0], st->dense1[1], 2 * XS_D1 * sizeof(float));
    dense1_frame(st, feats + 4 * XS_NF, per[4], st->dense1[2]);
    float cat[3 * XS_D1], c2[XS_D2], cond[4 * XS_C];
    memcpy(cat, st->dense1, sizeof cat);
    MV(st, fconv1, cat, c2); for (int i = 0; i < XS_D2; i++) c2[i] = tanh_(c2[i]);
    MV(st, fdense2, c2, cond); for (int i = 0; i < 4 * XS_C; i++) cond[i] = tanh_(cond[i]);
    int period = per[3];

    for (int k = 0; k < XS_NSUB; k++) {
        const float *cs = cond + k * XS_C;
        float g; mv(st, xs_cond_gain_w, xs_cond_gain_s, 1, XS_C, cs, &g); float gain = expf(g + xs_cond_gain_b[0]);
        float pred[XS_SUB + 4], prev[XS_SUB];
        for (int r = 0; r < XS_SUB + 4; r++) {
            int idx = 256 - period + r - 2; if (idx >= 256) idx -= period; if (idx < 0) idx = 0;
            pred[r] = clamp1(st->exc_mem[idx] / (1e-5f + gain));
        }
        for (int r = 0; r < XS_SUB; r++) prev[r] = clamp1(st->exc_mem[256 - XS_SUB + r] / (1e-5f + gain));
        const float *fp = pred + 2;
        /* fwc0 */
        float tmp[XS_FIN], xcat[2 * XS_FIN], fw[XS_F];
        memcpy(tmp, cs, XS_C * sizeof(float)); memcpy(tmp + XS_C, pred, (XS_SUB + 4) * sizeof(float)); memcpy(tmp + XS_C + XS_SUB + 4, prev, XS_SUB * sizeof(float));
        memcpy(xcat, st->fwc_state, XS_FIN * sizeof(float)); memcpy(xcat + XS_FIN, tmp, XS_FIN * sizeof(float));
        MV(st, fwc0, xcat, fw); for (int i = 0; i < XS_F; i++) fw[i] = tanh_(fw[i]);
        GLU(st, fwc0_glu, XS_F, fw); for (int i = 0; i < XS_F; i++) fw[i] = clamp1(fw[i]);
        memcpy(st->fwc_state, tmp, XS_FIN * sizeof(float));
        float pgl[4], pg[4]; MV(st, gain_out, fw, pgl); for (int i = 0; i < 4; i++) pg[i] = sigmoidf_(pgl[i] + xs_gain_out_b[i]);
        /* três GRUs com GLU */
        float in1[XS_F + 2 * XS_SUB], o1[XS_H1], in2[XS_H1 + 2 * XS_SUB], o2[XS_H2], in3[XS_H2 + 2 * XS_SUB], o3[XS_H3];
        memcpy(in1, fw, XS_F * sizeof(float)); for (int i = 0; i < XS_SUB; i++) { in1[XS_F + i] = pg[0] * fp[i]; in1[XS_F + XS_SUB + i] = prev[i]; }
        gru(st, xs_gru1_ih_w, xs_gru1_ih_s, xs_gru1_hh_w, xs_gru1_hh_s, XS_F + 2 * XS_SUB, XS_H1, in1, st->h1);
        for (int i = 0; i < XS_H1; i++) o1[i] = clamp1(st->h1[i]);
        GLU(st, gru1_glu, XS_H1, o1); for (int i = 0; i < XS_H1; i++) o1[i] = clamp1(o1[i]);
        memcpy(in2, o1, XS_H1 * sizeof(float)); for (int i = 0; i < XS_SUB; i++) { in2[XS_H1 + i] = pg[1] * fp[i]; in2[XS_H1 + XS_SUB + i] = prev[i]; }
        gru(st, xs_gru2_ih_w, xs_gru2_ih_s, xs_gru2_hh_w, xs_gru2_hh_s, XS_H1 + 2 * XS_SUB, XS_H2, in2, st->h2);
        for (int i = 0; i < XS_H2; i++) o2[i] = clamp1(st->h2[i]);
        GLU(st, gru2_glu, XS_H2, o2); for (int i = 0; i < XS_H2; i++) o2[i] = clamp1(o2[i]);
        memcpy(in3, o2, XS_H2 * sizeof(float)); for (int i = 0; i < XS_SUB; i++) { in3[XS_H2 + i] = pg[2] * fp[i]; in3[XS_H2 + XS_SUB + i] = prev[i]; }
        gru(st, xs_gru3_ih_w, xs_gru3_ih_s, xs_gru3_hh_w, xs_gru3_hh_s, XS_H2 + 2 * XS_SUB, XS_H3, in3, st->h3);
        for (int i = 0; i < XS_H3; i++) o3[i] = clamp1(st->h3[i]);
        GLU(st, gru3_glu, XS_H3, o3); for (int i = 0; i < XS_H3; i++) o3[i] = clamp1(o3[i]);
        /* skip */
        float sk[XS_H1 + XS_H2 + XS_H3 + XS_F + 2 * XS_SUB], so[XS_S], sig[XS_SUB]; int p = 0;
        memcpy(sk + p, o1, XS_H1 * sizeof(float)); p += XS_H1; memcpy(sk + p, o2, XS_H2 * sizeof(float)); p += XS_H2;
        memcpy(sk + p, o3, XS_H3 * sizeof(float)); p += XS_H3; memcpy(sk + p, fw, XS_F * sizeof(float)); p += XS_F;
        for (int i = 0; i < XS_SUB; i++) sk[p + i] = pg[3] * fp[i]; p += XS_SUB; memcpy(sk + p, prev, XS_SUB * sizeof(float));
        MV(st, skip_dense, sk, so); for (int i = 0; i < XS_S; i++) so[i] = clamp1(tanh_(so[i]));
        GLU(st, skip_glu, XS_S, so);
        MV(st, sig_out, so, sig);
        for (int i = 0; i < XS_SUB; i++) sig[i] = tanh_(sig[i]) * gain;
        memmove(st->exc_mem, st->exc_mem + XS_SUB, (256 - XS_SUB) * sizeof(float));
        memcpy(st->exc_mem + 256 - XS_SUB, sig, XS_SUB * sizeof(float));
        memcpy(out + k * XS_SUB, sig, XS_SUB * sizeof(float));
    }
}
