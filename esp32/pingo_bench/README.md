# pingo-ptbr edge no ESP32-S3: benchmark

Roda o pingo inteiro em C (acústico em **4 bits** + vocoder **xxs** int8, ativações int8, streaming) sobre uma frase fixa
("Olá, eu sou a voz da SemFence.", 30 fonemas, 2,29 s de áudio) e imprime pela serial o tempo de cada fase, o fator de tempo real,
os MAC, a RAM livre e o rms do áudio (para comparar com o do host).

**Estado: compila para esp32s3 e o mesmo código roda no Mac com um shim das APIs do ESP (229 quadros, rms 4203, igual à referência).
Nunca foi executado numa placa.** Todo número de velocidade e de heap abaixo é estimativa até alguém medir.
Se você rodar, abra uma issue com as linhas que ele imprime: é o dado que falta.

## Como usar

```bash
# ESP-IDF 5.x ou 6.x (https://docs.espressif.com/projects/esp-idf/)
. $IDF_PATH/export.sh
cd esp32/pingo_bench
idf.py set-target esp32s3
idf.py build flash monitor
```

Ou sem instalar nada, no Docker: `docker run --rm -v $PWD:/project -w /project espressif/idf:latest idf.py set-target esp32s3 build`.
Precisa de ≥ 4 MB de flash. PSRAM é opcional.

Placas com PSRAM (N8R2 etc.): `SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.psram" idf.py set-target esp32s3 build`
(chip octal, N16R8/N8R8: troque `CONFIG_SPIRAM_MODE_QUAD` por `CONFIG_SPIRAM_MODE_OCT=y` em `sdkconfig.psram`). Com PSRAM o encoder
vai para ela e o limite sobe de 64 para 256 tokens por frase; a mesma imagem roda em placas sem PSRAM (limite 64).
O cache de dados de 64 KB que `sdkconfig.psram` liga é uma hipótese de velocidade, a medir (custa SRAM interna).

## O que ele imprime

```
fonemas -> quadros (esperado 229) = 2.29 s de áudio
acústico: encoder X ms + decoder Y ms | vocoder Z ms | total W ms (streaming)
fator de tempo real: Nx   (>1 = mais rápido que a fala)
rms do PCM A (host: 4203) -> OK
RAM interna ... livre / menor livre desde o boot / maior bloco
```

Para ouvir o resultado, troque `PINGO_DUMP_PCM` para `1` em `main/main.c`, regrave e salve a serial:

```bash
idf.py monitor | tee log.txt        # espere aparecer "fim" e feche com Ctrl+]
python3 tools/pcm_from_serial.py log.txt saida.wav
```

## Tamanho (medido pelo `idf.py size`, build no Docker)

| | firmware | RAM interna fixa (de 342 KB) |
|---|---:|---:|
| sem PSRAM | 443 KB | 148 KB |
| com `sdkconfig.psram` | 448 KB | 151 KB |

- Pesos do vocoder xxs: 107 KiB int8 em `DRAM_ATTR` (relidos 400×/s, precisam estar na SRAM). Pesos do acústico: 131 KiB em 4 bits, na flash.
- Pico de RAM de trabalho (**estimativa**, não medida): ~50 KB para a frase de teste (encoder proporcional à frase: 23 KB para 30 tokens; cache de atenção do decoder 33 KB; pilha 16 KB).
- O ESP-IDF é a maior parte do binário: nossa parte (pesos + código) é ~427 KB.

## Variantes

- `variants/xs_weights.h`: vocoder xs (160 mil parâmetros, 56 MMAC/s). Troque `main/xs_weights.h` e ajuste `TS_PCM_RMS_EXPECTED`.
- `variants/ac_weights_8bit.h`: acústico em 8 bits (262 KiB). Com ele a frase de teste dura 248 quadros (em 4 bits, 229: nesta frase as durações
  previstas ficam 8% mais curtas; na média de 21 frases a diferença é de só −2,2%; o parâmetro de velocidade compensa).

## Se o fator de tempo real ficar abaixo de 1

Em ordem do mais simples ao mais trabalhoso (nada disso foi testado na placa):

1. Conferir 240 MHz e `-O3` (já estão em `sdkconfig.defaults` e `main/CMakeLists.txt`).
2. Trocar o laço `mv8` de `main/fargan_xs_q8.c` pelo kernel int8 do **ESP-NN**, que usa o SIMD do S3 (PIE). Maior ganho esperado.
3. Acústico num núcleo e vocoder no outro (dois `xTaskCreatePinnedToCore` e uma fila de quadros).
4. Esparsificar as GRUs e retreinar (~US$ 0,6 de GPU).

## Limites

- O fonemizador ([dizione-ptbr](https://github.com/SemFence/dizione-ptbr)) não está aqui: a frase de teste já vem como fonemas em `main/test_sentence.h`.
  Para outras frases, gere os tokens no Mac (`vocoder-train/c_xxs/text_to_in.py`) ou porte o fonemizador.
- Sem saída de áudio real (I2S/DAC): só mede e, opcionalmente, despeja o PCM pela serial.
- `main/acoustic.*`, `main/fargan_xs*`, `main/xs_weights.h`, `main/ac_weights.h` são gerados/copiados de `c/` (`export_c.py`, `export_acoustic_c.py`).
