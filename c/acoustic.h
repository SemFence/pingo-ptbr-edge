/* pingo acoustic model in plain C: per-row quantized weights (flash/const, 4 or 8 bits), float activations. No malloc.
 * Phonemized text (tokens, stress, voice) -> durations -> 20-float features per 10 ms frame, one frame at a time (streaming). */
#pragma once
#include "ac_weights.h"

#ifndef AC_MAXTOK
#define AC_MAXTOK 256           /* max tokens per sentence; the encoder's real RAM use is proportional to the sentence length (ac_enc_ws_bytes) */
#endif
#ifndef AC_PHONE_LIMIT
#define AC_PHONE_LIMIT 20       /* max frames per phone (pingo_mlx.PHONE_LIMIT) */
#define AC_FIRST_PHONE_LIMIT 17 /* and for the first phone of the sentence */
#endif
#define AC_W AC_CFG_WIDTH
#define AC_HD (AC_CFG_WIDTH / AC_CFG_HEADS)
#define AC_KVW (AC_CFG_KV_HEADS * AC_HD)
#define AC_WIN AC_CFG_WINDOW

typedef struct {                          /* encoder workspace: pointers into one block of ac_enc_ws_bytes(L) bytes (internal RAM or PSRAM) */
    float (*x)[AC_W], (*t)[AC_W], (*k)[AC_KVW], (*v)[AC_KVW]; int cap;
} ac_enc_ws;
static inline unsigned long ac_enc_ws_bytes(int L) { return sizeof(float) * (unsigned long)L * (2 * AC_W + 2 * AC_KVW); }
static inline void ac_enc_ws_bind(ac_enc_ws *ws, void *mem, int L) {   /* mem: aligned block of ac_enc_ws_bytes(L) bytes */
    float *p = (float *)mem; ws->x = (float (*)[AC_W])p; p += (long)L * AC_W; ws->t = (float (*)[AC_W])p; p += (long)L * AC_W;
    ws->k = (float (*)[AC_KVW])p; p += (long)L * AC_KVW; ws->v = (float (*)[AC_KVW])p; ws->cap = L;
}

typedef struct {                          /* streaming decoder state: sliding attention window */
    float k[AC_CFG_DECODER_LAYERS][AC_WIN][AC_KVW], v[AC_CFG_DECODER_LAYERS][AC_WIN][AC_KVW];
    long macs;
} ac_dec_state;

/* enc: output (L x AC_W), after text_norm; logdur: log1p(duration) per token. Returns 0 on success. */
int ac_encode(ac_enc_ws *ws, const int *tok, const int *stress, int voice, int L, float *enc, float *logdur, long *macs);
/* durations in frames: mask[i]=1 for phones (minimum duration 1); tok[i]==0 is padding (duration 0). Returns the total number of frames. */
int ac_durations(const float *logdur, const int *tok, const int *mask, int L, float speed, int *dur);
void ac_dec_init(ac_dec_state *st);
/* one frame: enc_row = enc[token of this frame], phase = 4 floats, pos = frame index in the sentence. out = 20 de-normalized features. */
void ac_dec_frame(ac_dec_state *st, const float *enc_row, const float *phase, int pos, float *out);
