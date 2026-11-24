#include <math.h>
#include <string.h>
#include "acoustic.h"

#if AC_WBITS == 4          /* pesos em 4 bits: dois por byte; w(p, idx) devolve o peso de índice plano idx */
typedef uint8_t ac_wt;
static inline float aw(const ac_wt *p, long idx) { uint8_t b = p[idx >> 1]; return (float)((idx & 1) ? (int8_t)b >> 4 : (int8_t)(b << 4) >> 4); }
#else
typedef int8_t ac_wt;
static inline float aw(const ac_wt *p, long idx) { return (float)p[idx]; }
#endif
typedef struct { const ac_wt *w; const float *s; int rows, cols; } lin_t;
#define L(n) { ac_##n##_w, ac_##n##_s, AC_##n##_ROWS, AC_##n##_COLS }
typedef struct { const float *n1, *n2; lin_t q, k, v, o, gate, up, down; } attn_t;
typedef struct { const float *n1, *n2; lin_t conv; } conv_t;
#define ATTN(p) { ac_##p##_norm1_weight, ac_##p##_norm2_weight, L(p##_attention_q_weight), L(p##_attention_k_weight), L(p##_attention_v_weight), \
                  L(p##_attention_o_weight), L(p##_ffn_gate_weight), L(p##_ffn_up_weight), L(p##_ffn_down_weight) }
#define CONV(p) { ac_##p##_norm1_weight, ac_##p##_norm2_weight, L(p##_conv_weight) }
static const attn_t ENC_ATT[3] = { ATTN(text_blocks_0), ATTN(text_blocks_2), ATTN(text_blocks_4) };
static const conv_t ENC_CONV[3] = { CONV(text_blocks_1), CONV(text_blocks_3), CONV(text_blocks_5) };
static const attn_t DEC[AC_CFG_DECODER_LAYERS] = { ATTN(decoder_0), ATTN(decoder_1) };
static const lin_t DUR_IN = L(durations_in_proj_weight), DUR_OUT = L(durations_out_weight), DUR_PW[2] = { L(durations_pw_0_weight), L(durations_pw_1_weight) },
                   DUR_DW[2] = { L(durations_dw_0_weight), L(durations_dw_1_weight) }, HEAD = L(head_weight), PHASE = L(phase_weight);
static const float *DUR_PW_B[2] = { ac_durations_pw_0_bias, ac_durations_pw_1_bias }, *DUR_DW_B[2] = { ac_durations_dw_0_bias, ac_durations_dw_1_bias },
                   *DUR_NORM[2] = { ac_durations_norms_0_weight, ac_durations_norms_1_weight };

static inline float sigm(float x) { return 1.f / (1.f + expf(-x)); }
static inline float silu(float x) { return x * sigm(x); }

static void mv(const lin_t *l, const float *x, float *y, long *macs) {
    for (int i = 0; i < l->rows; i++) {
        const long base = (long)i * l->cols; float a = 0.f;
        for (int j = 0; j < l->cols; j++) a += x[j] * aw(l->w, base + j);
        y[i] = a * l->s[i];
    }
    *macs += (long)l->rows * l->cols;
}
static void rmsnorm(const float *x, const float *w, float *y, int n) {
    float ss = 0.f; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.f / sqrtf(ss / n + 1e-6f); for (int i = 0; i < n; i++) y[i] = x[i] * r * w[i];
}
static void rope(float *x, int heads, int pos) {            /* x: heads x AC_HD; metades [0,hd/2) e [hd/2,hd) */
    int d = AC_HD / 2;
    for (int h = 0; h < heads; h++) for (int i = 0; i < d; i++) {
        float ang = (float)pos * powf(10000.f, -(float)i / d), c = cosf(ang), s = sinf(ang);
        float a = x[h * AC_HD + i], b = x[h * AC_HD + d + i];
        x[h * AC_HD + i] = a * c - b * s; x[h * AC_HD + d + i] = a * s + b * c;
    }
}
/* atenção de uma consulta (AC_HEADS cabeças) sobre n chaves; a chave t está na linha (j0+t) % mod de K/V (mod=0: linha j0+t). */
static void attend(const float *q, const float (*K)[AC_KVW], const float (*V)[AC_KVW], int j0, int n, int mod, float *ctx, long *macs) {
    float sc[AC_MAXTOK > AC_WIN ? AC_MAXTOK : AC_WIN]; const int rep = AC_CFG_HEADS / AC_CFG_KV_HEADS;
    for (int h = 0; h < AC_CFG_HEADS; h++) {
        const int kv = h / rep; float mx = -1e30f;
        for (int t = 0; t < n; t++) {
            int row = mod ? (j0 + t) % mod : j0 + t; float a = 0.f;
            for (int i = 0; i < AC_HD; i++) a += q[h * AC_HD + i] * K[row][kv * AC_HD + i];
            sc[t] = a * 0.25f; if (sc[t] > mx) mx = sc[t];
        }
        float z = 0.f; for (int t = 0; t < n; t++) { sc[t] = expf(sc[t] - mx); z += sc[t]; }
        for (int i = 0; i < AC_HD; i++) {
            float a = 0.f;
            for (int t = 0; t < n; t++) a += sc[t] * V[mod ? (j0 + t) % mod : j0 + t][kv * AC_HD + i];
            ctx[h * AC_HD + i] = a / z;
        }
    }
    *macs += (long)AC_CFG_HEADS * n * AC_HD * 2;
}
static void ffn_add(const attn_t *b, float *x, long *macs) {      /* x += down(silu(gate(n2(x))) * up(n2(x))) */
    float xn[AC_W], g[AC_CFG_FF], u[AC_CFG_FF], d[AC_W];
    rmsnorm(x, b->n2, xn, AC_W); mv(&b->gate, xn, g, macs); mv(&b->up, xn, u, macs);
    for (int i = 0; i < AC_CFG_FF; i++) g[i] = silu(g[i]) * u[i];
    mv(&b->down, g, d, macs); for (int i = 0; i < AC_W; i++) x[i] += d[i];
}

#define RES(t, c) (*((c) < AC_KVW ? &ws->k[t][c] : &ws->v[t][(c) - AC_KVW]))   /* 64 floats por token, emprestados de k e v (ociosos aqui) */

int ac_encode(ac_enc_ws *ws, const int *tok, const int *stress, int voice, int Ln, float *enc, float *logdur, long *macs) {
    if (Ln > ws->cap || Ln > AC_MAXTOK) return 1;
    long m0 = 0; if (!macs) macs = &m0;
    for (int t = 0; t < Ln; t++) for (int i = 0; i < AC_W; i++)
        ws->x[t][i] = aw(ac_tokens_weight_w, (long)tok[t] * AC_W + i) * ac_tokens_weight_s[tok[t]] + aw(ac_stress_weight_w, (long)stress[t] * AC_W + i) * ac_stress_weight_s[stress[t]]
                    + aw(ac_voices_weight_w, (long)voice * AC_W + i) * ac_voices_weight_s[voice];
    for (int blk = 0; blk < AC_CFG_TEXT_LAYERS; blk++) {
        if (blk % 2 == 0) {
            const attn_t *b = &ENC_ATT[blk / 2]; float xn[AC_W], q[AC_W], ctx[AC_W], o[AC_W];
            for (int t = 0; t < Ln; t++) {                                    /* passo 1: k e v de todos os tokens */
                rmsnorm(ws->x[t], b->n1, xn, AC_W); mv(&b->k, xn, ws->k[t], macs); mv(&b->v, xn, ws->v[t], macs); rope(ws->k[t], AC_CFG_KV_HEADS, t);
            }
            for (int t = 0; t < Ln; t++) {                                    /* passo 2: atenção bidirecional */
                rmsnorm(ws->x[t], b->n1, xn, AC_W); mv(&b->q, xn, q, macs); rope(q, AC_CFG_HEADS, t);
                attend(q, ws->k, ws->v, 0, Ln, 0, ctx, macs); mv(&b->o, ctx, o, macs);
                for (int i = 0; i < AC_W; i++) ws->x[t][i] += o[i];
                ffn_add(b, ws->x[t], macs);
            }
        } else {
            const conv_t *b = &ENC_CONV[blk / 2]; const int dil = 1 << (blk / 2), K = AC_CFG_KERNEL;
            for (int t = 0; t < Ln; t++) rmsnorm(ws->x[t], b->n1, ws->t[t], AC_W);
            for (int t = 0; t < Ln; t++) {
                float cat[AC_CFG_KERNEL * AC_W], y[AC_W], yn[AC_W];
                for (int k = 0; k < K; k++) { int s = t + (k - K / 2) * dil; for (int i = 0; i < AC_W; i++) cat[k * AC_W + i] = (s >= 0 && s < Ln) ? ws->t[s][i] : 0.f; }
                mv(&b->conv, cat, y, macs); rmsnorm(y, b->n2, yn, AC_W);
                for (int i = 0; i < AC_W; i++) y[i] = silu(yn[i]);
                for (int i = 0; i < AC_W; i++) enc[t * AC_W + i] = y[i];          /* guarda o termo residual fora de x para não ler x já alterado */
            }
            for (int t = 0; t < Ln; t++) for (int i = 0; i < AC_W; i++) ws->x[t][i] += enc[t * AC_W + i];
        }
    }
    for (int t = 0; t < Ln; t++) rmsnorm(ws->x[t], ac_text_norm_weight, enc + t * AC_W, AC_W);
    /* preditor de durações */
    float (*h)[AC_W] = ws->x, (*u)[AC_W] = ws->t;                             /* reaproveita a área de trabalho: x já foi copiado para enc */
    for (int t = 0; t < Ln; t++) { float tmp[AC_CFG_DUR_WIDTH]; mv(&DUR_IN, enc + t * AC_W, tmp, macs); for (int i = 0; i < AC_CFG_DUR_WIDTH; i++) h[t][i] = silu(tmp[i] + ac_durations_in_proj_bias[i]); }
    for (int l = 0; l < 2; l++) {
        for (int t = 0; t < Ln; t++) rmsnorm(h[t], DUR_NORM[l], u[t], AC_CFG_DUR_WIDTH);
        for (int t = 0; t < Ln; t++) {
            float dw[AC_CFG_DUR_WIDTH], pw[AC_CFG_DUR_WIDTH];
            for (int c = 0; c < AC_CFG_DUR_WIDTH; c++) {
                float a = DUR_DW_B[l][c];
                for (int k = 0; k < 3; k++) { int s = t + k - 1; if (s >= 0 && s < Ln) a += aw(DUR_DW[l].w, (long)c * 3 + k) * DUR_DW[l].s[c] * u[s][c]; }
                dw[c] = a;
            }
            *macs += (long)AC_CFG_DUR_WIDTH * 3;
            mv(&DUR_PW[l], dw, pw, macs);
            for (int c = 0; c < AC_CFG_DUR_WIDTH; c++) RES(t, c) = silu(pw[c] + DUR_PW_B[l][c]);   /* termo residual: h só muda depois, o dw dos vizinhos ainda lê u */
        }
        for (int t = 0; t < Ln; t++) for (int c = 0; c < AC_CFG_DUR_WIDTH; c++) h[t][c] += RES(t, c);
    }
    for (int t = 0; t < Ln; t++) { float o; mv(&DUR_OUT, h[t], &o, macs); logdur[t] = o + ac_durations_out_bias[0]; }
    return 0;
}

int ac_durations(const float *logdur, const int *tok, const int *mask, int Ln, float speed, int *dur) {
    int total = 0, first = -1; const float cap = logf(1.f + AC_CFG_MAX_TOKEN_FRAMES);
    for (int i = 0; i < Ln; i++) if (mask[i]) { first = i; break; }
    int lim = (int)rintf((float)AC_PHONE_LIMIT / speed), flim = (int)rintf((float)AC_FIRST_PHONE_LIMIT / speed); if (lim < 1) lim = 1; if (flim < 1) flim = 1;
    for (int i = 0; i < Ln; i++) {
        float v = logdur[i]; v = v < 0.f ? 0.f : (v > cap ? cap : v);
        int d = (int)rintf(expm1f(v) / speed);
        if (mask[i]) { int c2 = (i == first) ? flim : lim; if (d > c2) d = c2; }   /* limite de duração por fonema (igual ao pingo_mlx) */
        if (mask[i] && d < 1) d = 1;
        if (d > AC_CFG_MAX_TOKEN_FRAMES) d = AC_CFG_MAX_TOKEN_FRAMES;
        if (tok[i] == 0) d = 0;
        dur[i] = d; total += d;
    }
    return total;
}

void ac_dec_init(ac_dec_state *st) { memset(st, 0, sizeof *st); }

void ac_dec_frame(ac_dec_state *st, const float *enc_row, const float *phase, int pos, float *out) {
    float x[AC_W], ph[AC_W]; long *m = &st->macs;
    mv(&PHASE, phase, ph, m); for (int i = 0; i < AC_W; i++) x[i] = enc_row[i] + ph[i];
    for (int l = 0; l < AC_CFG_DECODER_LAYERS; l++) {
        const attn_t *b = &DEC[l]; float xn[AC_W], q[AC_W], ctx[AC_W], o[AC_W], k[AC_KVW], v[AC_KVW];
        rmsnorm(x, b->n1, xn, AC_W); mv(&b->q, xn, q, m); mv(&b->k, xn, k, m); mv(&b->v, xn, v, m);
        if (l == 0) { rope(q, AC_CFG_HEADS, pos); rope(k, AC_CFG_KV_HEADS, pos); }
        memcpy(st->k[l][pos % AC_WIN], k, sizeof k); memcpy(st->v[l][pos % AC_WIN], v, sizeof v);
        int n = pos + 1 < AC_WIN ? pos + 1 : AC_WIN, j0 = pos + 1 - n;
        attend(q, st->k[l], st->v[l], j0, n, AC_WIN, ctx, m); mv(&b->o, ctx, o, m);
        for (int i = 0; i < AC_W; i++) x[i] += o[i];
        ffn_add(b, x, m);
    }
    float xn[AC_W], f[AC_CFG_FEATURE_DIM]; rmsnorm(x, ac_feature_norm_weight, xn, AC_W); mv(&HEAD, xn, f, m);
    for (int i = 0; i < AC_CFG_FEATURE_DIM; i++) out[i] = f[i] * ac_f_std[i] + ac_f_mean[i];
}
