/* Acústico do pingo em C puro: pesos int8 por linha (em flash/const), ativações em float. Sem malloc.
 * Texto fonemizado (tokens, stress, voz) -> durações -> features de 20 floats por quadro de 10 ms, um quadro por vez (streaming). */
#pragma once
#include "ac_weights.h"

#ifndef AC_MAXTOK
#define AC_MAXTOK 256           /* teto de tokens por frase; o uso real de RAM do encoder é proporcional ao tamanho da frase (ac_enc_ws_bytes) */
#endif
#ifndef AC_PHONE_LIMIT
#define AC_PHONE_LIMIT 20       /* quadros máx. por fonema (pingo_mlx.PHONE_LIMIT) */
#define AC_FIRST_PHONE_LIMIT 17 /* e para o primeiro fonema da frase */
#endif
#define AC_W AC_CFG_WIDTH
#define AC_HD (AC_CFG_WIDTH / AC_CFG_HEADS)
#define AC_KVW (AC_CFG_KV_HEADS * AC_HD)
#define AC_WIN AC_CFG_WINDOW

typedef struct {                          /* área de trabalho do encoder: ponteiros para um bloco de ac_enc_ws_bytes(L) bytes (RAM interna ou PSRAM) */
    float (*x)[AC_W], (*t)[AC_W], (*k)[AC_KVW], (*v)[AC_KVW]; int cap;
} ac_enc_ws;
static inline unsigned long ac_enc_ws_bytes(int L) { return sizeof(float) * (unsigned long)L * (2 * AC_W + 2 * AC_KVW); }
static inline void ac_enc_ws_bind(ac_enc_ws *ws, void *mem, int L) {   /* mem: bloco alinhado de ac_enc_ws_bytes(L) bytes */
    float *p = (float *)mem; ws->x = (float (*)[AC_W])p; p += (long)L * AC_W; ws->t = (float (*)[AC_W])p; p += (long)L * AC_W;
    ws->k = (float (*)[AC_KVW])p; p += (long)L * AC_KVW; ws->v = (float (*)[AC_KVW])p; ws->cap = L;
}

typedef struct {                          /* estado do decoder em streaming: janela de atenção deslizante */
    float k[AC_CFG_DECODER_LAYERS][AC_WIN][AC_KVW], v[AC_CFG_DECODER_LAYERS][AC_WIN][AC_KVW];
    long macs;
} ac_dec_state;

/* enc: saída (L x AC_W), vinda do text_norm; logdur: log1p(duração) por token. Retorna 0 se ok. */
int ac_encode(ac_enc_ws *ws, const int *tok, const int *stress, int voice, int L, float *enc, float *logdur, long *macs);
/* durações em quadros: mask[i]=1 para fonemas (duração mínima 1); tok[i]==0 é preenchimento (duração 0). Retorna o total de quadros. */
int ac_durations(const float *logdur, const int *tok, const int *mask, int L, float speed, int *dur);
void ac_dec_init(ac_dec_state *st);
/* um quadro: enc_row = enc[token do quadro], phase = 4 floats, pos = índice do quadro na frase. out = 20 features já desnormalizadas. */
void ac_dec_frame(ac_dec_state *st, const float *enc_row, const float *phase, int pos, float *out);
