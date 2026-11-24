#include <math.h>
#include <string.h>
#include "fargan_xs.h"
/* Variante com ATIVAÇÕES int8 (escala fixa 1/127, como o dither n() do treino) e acumulação int32. Pesos int8. */

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


static inline int8_t q8(float x) { float v = x * 127.f; v = v > 127.f ? 127.f : (v < -127.f ? -127.f : v); return (int8_t)lrintf(v); }
#define INV127 (1.0f / 127.0f)

/* y[i] = escala[i]/127 * sum_j w[i][j] * xq[j]    (int8 x int8 -> int32) */
static void mv8(xs_state *st, const int8_t *w, const float *s, int rows, int cols, const int8_t *xq, float *y) {
    for (int i = 0; i < rows; i++) {
        const int8_t *r = w + (long)i * cols; int32_t acc = 0;
        for (int j = 0; j < cols; j++) acc += (int32_t)xq[j] * (int32_t)r[j];
        y[i] = (float)acc * s[i] * INV127;
    }
    st->macs += (long)rows * cols;
}
#define MV8(st, name, xq, y) mv8(st, xs_##name##_w, xs_##name##_s, XS_##name##_ROWS, XS_##name##_COLS, xq, y)

void xs_init(xs_state *st) { memset(st, 0, sizeof *st); }

static int period_of(float f18) {
    float p = f18 < -10.f ? -10.f : (f18 > 10.f ? 10.f : f18);
    int v = (int)rintf(256.f / exp2f(p + 1.5f));
    return v < 32 ? 32 : (v > 255 ? 255 : v);
}

static void dense1_frame(xs_state *st, const float *f, int per, float *out) {      /* entrada crua (não limitada): fica em float */
    float in[XS_NF + XS_PE];
    memcpy(in, f, XS_NF * sizeof(float));
    const int8_t *e = xs_pembed_w + (long)(per - 32) * XS_pembed_COLS;
    for (int i = 0; i < XS_PE; i++) in[XS_NF + i] = (float)e[i] * xs_pembed_s[per - 32];
    mv(st, xs_fdense1_w, xs_fdense1_s, XS_fdense1_ROWS, XS_fdense1_COLS, in, out);
    for (int i = 0; i < XS_D1; i++) out[i] = tanh_(out[i]);
}

/* x (float, já limitado a [-1,1]) -> y = x * sigmoid(W x) */
static void glu8(xs_state *st, const int8_t *w, const float *s, int n, float *x, const int8_t *xq) {
    float g[256]; mv8(st, w, s, n, n, xq, g);
    for (int i = 0; i < n; i++) x[i] *= sigmoidf_(g[i]);
}
#define GLU8(st, name, n, x, xq) glu8(st, xs_##name##_w, xs_##name##_s, n, x, xq)

static void gru8(xs_state *st, const int8_t *wi, const float *si, const int8_t *wh, const float *sh, int nin, int nh, const int8_t *xq, float *h) {
    float gi[3 * 64 * 3], gh[3 * 64 * 3]; int8_t hq[XS_H1 > XS_H2 ? XS_H1 : XS_H2];
    for (int i = 0; i < nh; i++) hq[i] = q8(h[i]);
    mv8(st, wi, si, 3 * nh, nin, xq, gi); mv8(st, wh, sh, 3 * nh, nh, hq, gh);
    for (int i = 0; i < nh; i++) {
        float r = sigmoidf_(gi[i] + gh[i]), z = sigmoidf_(gi[nh + i] + gh[nh + i]);
        float c = tanh_(gi[2 * nh + i] + r * gh[2 * nh + i]);
        h[i] = (1.f - z) * c + z * h[i];
    }
}
static inline float cl(float x) { return clamp1(x); }

void xs_frame(xs_state *st, const float *feats, float *out) {
    int per[5]; for (int j = 0; j < 5; j++) per[j] = period_of(feats[j * XS_NF + 18]);
    if (!st->primed) { for (int j = 0; j < 2; j++) dense1_frame(st, feats + (2 + j) * XS_NF, per[2 + j], st->dense1[j + 1]); st->primed = 1; }
    memmove(st->dense1[0], st->dense1[1], 2 * XS_D1 * sizeof(float));
    dense1_frame(st, feats + 4 * XS_NF, per[4], st->dense1[2]);
    int8_t catq[3 * XS_D1], c2q[XS_D2]; float c2[XS_D2], cond[4 * XS_C];
    for (int i = 0; i < 3 * XS_D1; i++) catq[i] = q8(((float *)st->dense1)[i]);
    MV8(st, fconv1, catq, c2); for (int i = 0; i < XS_D2; i++) c2q[i] = q8(tanh_(c2[i]));
    MV8(st, fdense2, c2q, cond); for (int i = 0; i < 4 * XS_C; i++) cond[i] = tanh_(cond[i]);
    int period = per[3];

    for (int k = 0; k < XS_NSUB; k++) {
        const float *cs = cond + k * XS_C;
        int8_t csq[XS_C]; for (int i = 0; i < XS_C; i++) csq[i] = q8(cs[i]);
        float g; mv8(st, xs_cond_gain_w, xs_cond_gain_s, 1, XS_C, csq, &g); float gain = expf(g + xs_cond_gain_b[0]);
        int8_t predq[XS_SUB + 4], prevq[XS_SUB];
        for (int r = 0; r < XS_SUB + 4; r++) {
            int idx = 256 - period + r - 2; if (idx >= 256) idx -= period; if (idx < 0) idx = 0;
            predq[r] = q8(st->exc_mem[idx] / (1e-5f + gain));
        }
        for (int r = 0; r < XS_SUB; r++) prevq[r] = q8(st->exc_mem[256 - XS_SUB + r] / (1e-5f + gain));
        const int8_t *fpq = predq + 2;
        int8_t tmpq[XS_FIN], xcatq[2 * XS_FIN], fwq[XS_F]; float fw[XS_F];
        memcpy(tmpq, csq, XS_C); memcpy(tmpq + XS_C, predq, XS_SUB + 4); memcpy(tmpq + XS_C + XS_SUB + 4, prevq, XS_SUB);
        memcpy(xcatq, st->fwc_q, XS_FIN); memcpy(xcatq + XS_FIN, tmpq, XS_FIN);
        MV8(st, fwc0, xcatq, fw); for (int i = 0; i < XS_F; i++) { fw[i] = tanh_(fw[i]); fwq[i] = q8(fw[i]); }
        glu8(st, xs_fwc0_glu_w, xs_fwc0_glu_s, XS_F, fw, fwq); for (int i = 0; i < XS_F; i++) { fw[i] = cl(fw[i]); fwq[i] = q8(fw[i]); }
        memcpy(st->fwc_q, tmpq, XS_FIN);
        float pgl[4], pg[4]; MV8(st, gain_out, fwq, pgl); for (int i = 0; i < 4; i++) pg[i] = sigmoidf_(pgl[i] + xs_gain_out_b[i]);
        int8_t pfq[4][XS_SUB];
        for (int j = 0; j < 4; j++) for (int i = 0; i < XS_SUB; i++) pfq[j][i] = q8(pg[j] * (float)fpq[i] * INV127);
        /* três GRUs com GLU */
        int8_t in1[XS_F + 2 * XS_SUB], in2[XS_H1 + 2 * XS_SUB], in3[XS_H2 + 2 * XS_SUB], o1q[XS_H1], o2q[XS_H2], o3q[XS_H3]; float o1[XS_H1], o2[XS_H2], o3[XS_H3];
        memcpy(in1, fwq, XS_F); memcpy(in1 + XS_F, pfq[0], XS_SUB); memcpy(in1 + XS_F + XS_SUB, prevq, XS_SUB);
        gru8(st, xs_gru1_ih_w, xs_gru1_ih_s, xs_gru1_hh_w, xs_gru1_hh_s, XS_F + 2 * XS_SUB, XS_H1, in1, st->h1);
        for (int i = 0; i < XS_H1; i++) { o1[i] = cl(st->h1[i]); o1q[i] = q8(o1[i]); }
        glu8(st, xs_gru1_glu_w, xs_gru1_glu_s, XS_H1, o1, o1q); for (int i = 0; i < XS_H1; i++) { o1[i] = cl(o1[i]); o1q[i] = q8(o1[i]); }
        memcpy(in2, o1q, XS_H1); memcpy(in2 + XS_H1, pfq[1], XS_SUB); memcpy(in2 + XS_H1 + XS_SUB, prevq, XS_SUB);
        gru8(st, xs_gru2_ih_w, xs_gru2_ih_s, xs_gru2_hh_w, xs_gru2_hh_s, XS_H1 + 2 * XS_SUB, XS_H2, in2, st->h2);
        for (int i = 0; i < XS_H2; i++) { o2[i] = cl(st->h2[i]); o2q[i] = q8(o2[i]); }
        glu8(st, xs_gru2_glu_w, xs_gru2_glu_s, XS_H2, o2, o2q); for (int i = 0; i < XS_H2; i++) { o2[i] = cl(o2[i]); o2q[i] = q8(o2[i]); }
        memcpy(in3, o2q, XS_H2); memcpy(in3 + XS_H2, pfq[2], XS_SUB); memcpy(in3 + XS_H2 + XS_SUB, prevq, XS_SUB);
        gru8(st, xs_gru3_ih_w, xs_gru3_ih_s, xs_gru3_hh_w, xs_gru3_hh_s, XS_H2 + 2 * XS_SUB, XS_H3, in3, st->h3);
        for (int i = 0; i < XS_H3; i++) { o3[i] = cl(st->h3[i]); o3q[i] = q8(o3[i]); }
        glu8(st, xs_gru3_glu_w, xs_gru3_glu_s, XS_H3, o3, o3q); for (int i = 0; i < XS_H3; i++) { o3[i] = cl(o3[i]); o3q[i] = q8(o3[i]); }
        /* skip */
        int8_t skq[XS_H1 + XS_H2 + XS_H3 + XS_F + 2 * XS_SUB], soq[XS_S]; float so[XS_S], sig[XS_SUB]; int p = 0;
        memcpy(skq + p, o1q, XS_H1); p += XS_H1; memcpy(skq + p, o2q, XS_H2); p += XS_H2; memcpy(skq + p, o3q, XS_H3); p += XS_H3;
        memcpy(skq + p, fwq, XS_F); p += XS_F; memcpy(skq + p, pfq[3], XS_SUB); p += XS_SUB; memcpy(skq + p, prevq, XS_SUB);
        MV8(st, skip_dense, skq, so); for (int i = 0; i < XS_S; i++) { so[i] = clamp1(tanh_(so[i])); soq[i] = q8(so[i]); }
        glu8(st, xs_skip_glu_w, xs_skip_glu_s, XS_S, so, soq); for (int i = 0; i < XS_S; i++) soq[i] = q8(so[i]);
        MV8(st, sig_out, soq, sig);
        for (int i = 0; i < XS_SUB; i++) sig[i] = tanh_(sig[i]) * gain;
        memmove(st->exc_mem, st->exc_mem + XS_SUB, (256 - XS_SUB) * sizeof(float));
        memcpy(st->exc_mem + 256 - XS_SUB, sig, XS_SUB * sizeof(float));
        memcpy(out + k * XS_SUB, sig, XS_SUB * sizeof(float));
    }
}
