/* FARGAN reduzido (xs) em C puro: pesos int8 por linha, ativações em float. Sem malloc, sem dependências além de libm.
 * Uso: xs_init(&st); para cada quadro n: xs_frame(&st, feats + 20*n, out160)  (feats = quadros n..n+4, 20 floats cada).
 * Para o quadro n sair alinhado a feats[n], o chamador entra com [3 cópias do 1º quadro] + feats + [1 cópia do último]. */
#pragma once
#include <stdint.h>
#include "xs_weights.h"

#define XS_NF 20            /* floats por quadro de features */
#define XS_PE 12            /* dimensão do embedding de pitch */
#define XS_SUB 40           /* amostras por subframe */
#define XS_NSUB 4
#define XS_FRAME (XS_SUB * XS_NSUB)
#define XS_FIN (2 * XS_SUB + XS_C + 4)   /* entrada do fwc0 (um passo) */

typedef struct {
    float exc_mem[256];
    float h1[XS_H1], h2[XS_H2], h3[XS_H3];
    float fwc_state[XS_FIN];
    int8_t fwc_q[XS_FIN];            /* só a variante com ativações int8 usa este estado */
    float dense1[3][XS_D1];          /* anel com a saída do fdense1 dos 3 últimos quadros */
    int primed;
    long macs;                       /* contador de multiplicações-acumulações (só para medir) */
} xs_state;

void xs_init(xs_state *st);
/* feats: 5 quadros consecutivos (n .. n+4) x 20 floats. out: 160 amostras de excitação (antes da de-ênfase 0,85). */
void xs_frame(xs_state *st, const float *feats, float *out);
