/* pingo-ptbr edge (acústico 4 bits + vocoder xxs int8, ativações int8) no ESP32-S3: benchmark.
 * Roda a frase de teste N vezes e imprime pela serial: tempo de cada fase, fator de tempo real, MAC, RAM livre e o rms do áudio
 * (para comparar com o do host). Para ouvir o resultado: defina PINGO_DUMP_PCM 1 e use tools/pcm_from_serial.py. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "acoustic.h"
#include "fargan_xs.h"
#include "test_sentence.h"

#define PINGO_RUNS 3
#define PINGO_DUMP_PCM 0     /* 1 = imprime o PCM em hexadecimal (lento: ~1 s a mais por segundo de áudio) */
static const char *TAG = "pingo";

static void *ialloc(size_t n) { void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT); if (!p) ESP_LOGE(TAG, "sem RAM interna para %u bytes", (unsigned)n); return p; }

/* buffers percorridos em sequência (encoder, features, saída) vão para a PSRAM se existir; senão ficam na RAM interna.
 * Pesos do vocoder e cache KV do decoder (relidos centenas de vezes por segundo) ficam sempre na RAM interna. */
static void *balloc(size_t n) {
    void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    return p ? p : ialloc(n);
}

static void report_heap(const char *when) {
    ESP_LOGI(TAG, "RAM interna %-22s livre %6u B | menor livre desde o boot %6u B | maior bloco %6u B", when,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
}

static int run_once(int dump) {
    int64_t t0 = esp_timer_get_time();
    /* limite de tokens: com PSRAM o encoder (proporcional à frase) vai para ela e frases de até AC_MAXTOK cabem; sem PSRAM, 64 tokens (~49 KB) */
    int maxtok = heap_caps_get_total_size(MALLOC_CAP_SPIRAM) ? AC_MAXTOK : 64;
    if (TS_L > maxtok) { ESP_LOGE(TAG, "frase com %d tokens, máximo %d nesta placa", TS_L, maxtok); return -1; }
    ac_enc_ws wsv, *ws = &wsv; void *wsmem = balloc(ac_enc_ws_bytes(TS_L)); if (wsmem) ac_enc_ws_bind(ws, wsmem, TS_L); float *enc = balloc(sizeof(float) * TS_L * AC_W), *ld = balloc(sizeof(float) * TS_L);
    int *dur = balloc(sizeof(int) * TS_L); long enc_macs = 0;
    if (!wsmem || !enc || !ld || !dur) return -1;
    if (ac_encode(ws, ts_tok, ts_stress, TS_VOICE, TS_L, enc, ld, &enc_macs)) { ESP_LOGE(TAG, "frase longa demais"); return -1; }
    free(wsmem);                                                /* o encoder acabou: devolve a memória antes do decoder */
    int T = ac_durations(ld, ts_tok, ts_mask, TS_L, 1.0f, dur);
    int64_t t1 = esp_timer_get_time();
    /* streaming: o acústico entrega um quadro por vez e o vocoder consome com 1 quadro de atraso; só uma janela de 5 quadros fica em memória */
    ac_dec_state *ds = ialloc(sizeof *ds); xs_state *vs = ialloc(sizeof *vs);
    if (!ds || !vs) return -1;
    ac_dec_init(ds); xs_init(vs);
    float hist[5 * XS_NF], fr0[XS_NF], ex[XS_FRAME], y = 0.f, ss = 0.f; long ns = 0; int t = 0, n = 0; int64_t t_ac = 0, t_vo = 0;
    #define EMIT() do { int64_t a0 = esp_timer_get_time(); xs_frame(vs, hist, ex); t_vo += esp_timer_get_time() - a0; n++; \
        for (int i = 0; i < XS_FRAME; i++) { y = ex[i] + 0.85f * y; float v = y * 32768.f; v = v > 32767.f ? 32767.f : (v < -32768.f ? -32768.f : v); \
            ss += v * v; ns++; if (dump) printf("%04x", (unsigned)(int16_t)lrintf(v) & 0xffff); } \
        if (dump && n % 10 == 0) printf("\n"); } while (0)
    for (int i = 0; i < TS_L; i++) for (int k = 0; k < dur[i]; k++, t++) {
        float fr = ((float)k + 0.5f) / (float)dur[i], ph[4] = { 2 * fr - 1, sinf(3.14159265f * fr), cosf(3.14159265f * fr), logf(1.f + dur[i]) / logf(1.f + AC_CFG_MAX_TOKEN_FRAMES) };
        int64_t a0 = esp_timer_get_time(); ac_dec_frame(ds, enc + i * AC_W, ph, t, fr0); t_ac += esp_timer_get_time() - a0;
        if (t == 0) { for (int j = 0; j < 5; j++) memcpy(hist + j * XS_NF, fr0, sizeof fr0); }
        else { memmove(hist, hist + XS_NF, 4 * XS_NF * sizeof(float)); memcpy(hist + 4 * XS_NF, fr0, sizeof fr0); EMIT(); }
    }
    memmove(hist, hist + XS_NF, 4 * XS_NF * sizeof(float)); memcpy(hist + 4 * XS_NF, fr0, sizeof fr0); EMIT();   /* último quadro: repete o final */
    long dec_macs = ds->macs, voc_macs = vs->macs; free(ds); free(vs); free(enc); free(ld); free(dur);
    int64_t t3 = esp_timer_get_time(), t2 = t1 + t_ac;
    (void)t2;
    double secs = T / 100.0, tac = (t1 - t0 + t_ac) / 1e6, tvo = t_vo / 1e6;
    ESP_LOGI(TAG, "%d fonemas -> %d quadros (esperado %d) = %.2f s de áudio", TS_L, T, TS_FRAMES_EXPECTED, secs);
    ESP_LOGI(TAG, "acústico: encoder %.1f ms + decoder %.1f ms | vocoder %.1f ms | total %.1f ms (streaming)", (t1 - t0) / 1e3, t_ac / 1e3, t_vo / 1e3, (t3 - t0) / 1e3);
    ESP_LOGI(TAG, "MAC: acústico %.1f M + vocoder %.1f M = %.1f M por %.2f s de áudio", (enc_macs + dec_macs) / 1e6, voc_macs / 1e6, (enc_macs + dec_macs + voc_macs) / 1e6, secs);
    ESP_LOGI(TAG, "fator de tempo real: %.2fx (>1 = mais rápido que a fala) | vocoder sozinho %.2fx | %.1f MMAC/s efetivos", secs / (tac + tvo), secs / tvo, (enc_macs + dec_macs + voc_macs) / 1e6 / (tac + tvo));
    double rms = sqrt(ss / (double)ns);
    ESP_LOGI(TAG, "rms do PCM %.0f (host: %.0f) -> %s", rms, (double)TS_PCM_RMS_EXPECTED, fabs(rms / TS_PCM_RMS_EXPECTED - 1) < 0.08 ? "OK" : "DIFERENTE: conferir");
    return 0;
}

static void task(void *arg) {
    ESP_LOGI(TAG, "CPU %d MHz | vocoder xxs int8 (pesos em DRAM) + acústico 4 bits (pesos na flash)", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    ESP_LOGI(TAG, "PSRAM: %u B livres (0 = placa sem PSRAM, tudo na RAM interna)", (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    report_heap("no início");
    for (int r = 0; r < PINGO_RUNS; r++) { ESP_LOGI(TAG, "--- rodada %d ---", r + 1); if (run_once(PINGO_DUMP_PCM && r == PINGO_RUNS - 1)) break; report_heap("após a rodada"); }
    ESP_LOGI(TAG, "fim");
    vTaskDelete(NULL);
}

void app_main(void) { xTaskCreatePinnedToCore(task, "pingo", 16384, NULL, 5, NULL, 1); }
