# pingo-ptbr-edge

Modelo e pesos também em / model and weights also on Hugging Face: <https://huggingface.co/semfence/pingo-ptbr-edge>

A edição do [pingo-ptbr](https://huggingface.co/semfence/pingo-ptbr) feita para **microcontrolador** (ESP32-S3): **379.814 parâmetros** no total, em C puro (só libm), sem PyTorch e sem rede.
O modelo acústico (270.081 parâmetros) guarda os pesos em **4 bits**; o vocoder é um FARGAN reduzido, o **xxs, com 109.733 parâmetros** (int8, 107 KiB), treinado do zero.
O pingo original tem 1.086.678 parâmetros, dos quais 817 mil são o vocoder. Aqui o vocoder encolheu 7,4× e custa **~39,6 MMAC/s em vez de 299**.

> **Status honesto.** O firmware **compila para esp32s3** e o mesmo código roda no Mac com a mesma saída da referência, mas **nunca foi executado numa placa**.
> Não há medida de tempo real no ESP32-S3, de uso real de RAM nem de WER desta edição. O que está medido e o que é estimativa está separado abaixo.
> Se você rodar numa placa, o que `esp32/pingo_bench` imprime pela serial é exatamente o dado que falta.

## Ouça

Amostras geradas **pelo código em C** (`c/`), sem edição, voz F_s4:

| Texto | Áudio |
|---|---|
| Olá, eu sou a voz da SemFence. | [ouvir / listen](samples/ola_semfence_edge.wav) |
| Sistema iniciado. Pingo pronto para falar. | [ouvir / listen](samples/startup_edge.wav) |

Em `samples/vocoder_xs_vs_xxs/` há as mesmas frases com os vocoders xs e xxs (int8), e em `samples/acoustic_bits/` o acústico em float, 8, 6, 5 e 4 bits.
A escuta foi feita pelo autor (uma pessoa), não é um teste com ouvintes. O som é o do pingo: de "chamada telefônica", um pouco metálico.

## O que mudou em relação ao pingo-ptbr

| | pingo-ptbr | pingo-ptbr-edge |
|---|---:|---:|
| Acústico | 270.081 par., 5 a 8 bits | 270.081 par., **4 bits** (131 KiB) |
| Vocoder | FARGAN 816.597 par. (do PiVo) | **xxs 109.733 par.**, int8 (107 KiB), treinado do zero |
| Custo do vocoder | ~299 MMAC/s | **~39,6 MMAC/s** |
| Runtime | MLX / ONNX | **C puro** (ativações int8, matemática racional) |
| Alvo | Mac, navegador | **ESP32-S3** (e qualquer CPU com C) |

Variantes intermediárias que também foram treinadas e estão em `training/sizes.py`: **s** (262 mil par., 90 MMAC/s) e **xs** (160 mil par., 56 MMAC/s).

## O vocoder xxs

FARGAN com camadas mais estreitas (`C=24, d1=32, d2=48, F=48, H1=48, H2=40, H3=40, S=40`). Treino na receita do Opus (perda STFT multirresolução + 0,03 × cosseno,
AdamW, quantização suave de 8 bits durante o treino) sobre 500 mil sequências de fala sintética (o dobro do xs), 45 épocas (melhor: 38), uma RTX 3090 alugada por 133 min (~US$ 0,5).

Perda espectral na validação (`val_spect`, menor é melhor; mesmo conjunto, mesmas características):

| | parâmetros | MMAC/s | val_spect |
|---|---:|---:|---:|
| s | 262 mil | 90 | 0,2487 |
| **xxs** (float) | 110 mil | 38,7 | **0,2656** |
| xxs (int8 simulado) | 110 mil | 38,7 | 0,2661 |
| xs | 160 mil | 56 | 0,2814 |

Cuidado ao ler: o xxs foi treinado com o **dobro de dados** do xs (500 mil sequências contra 250 mil), então ele ficar à frente do xs **não** prova que a arquitetura é melhor.
A perda espectral também não é qualidade percebida; a decisão do autor pelo xxs foi de ouvido.

## Acústico em 4 bits

Quantização pós-treino, simétrica, por linha (valores −7..7, escala float por linha), dois pesos por byte. Apenas os pesos 2D são quantizados; vetores ficam em float.
Efeito colateral medido: as **durações previstas ficam ~8% mais curtas** (229 contra 248 quadros na frase de teste), então a fala sai um pouco mais rápida; o parâmetro de velocidade compensa.
Em 5 bits as durações sobem 2 a 5%, em 6 bits caem ~1,5%, em 8 bits ±0,6%. Por isso a comparação objetiva das características não é direta e a escolha de 4 bits foi de ouvido.

## Tamanho no ESP32-S3

Medido pelo `idf.py size` (ESP-IDF 6.x no Docker, esp32s3, enxuto em `sdkconfig.defaults`):

| | firmware | RAM interna fixa (de 342 KB) |
|---|---:|---:|
| sem PSRAM | 443 KB | 148 KB |
| com PSRAM (`sdkconfig.psram`) | 448 KB | 151 KB |

Pesos do vocoder em SRAM interna (relidos 400×/s); pesos do acústico na flash. O encoder usa memória proporcional à frase (23 KB para 30 tokens) e, havendo PSRAM, vai para ela e o limite sobe de 64 para 256 tokens.
**Estimativa (não medida na placa):** pico de RAM interna de trabalho ~50 KB na frase de teste.

## O que está medido e o que não está

Medido (no Mac ou no build):
- Parâmetros, MMAC/s, tamanhos de firmware e de RAM fixa (acima).
- O acústico em C bate com a referência MLX (mesmos pesos arredondados): erro rms relativo 0,0000% em 30 tokens / 229 quadros.
- O vocoder em C (float) bate com o PyTorch (erro 1e-6). A versão int8 com ativações int8 diverge amostra a amostra do PyTorch (o vocoder é recursivo), então foi validada por rms e escuta (rms 4577 contra 4604 do xs na mesma frase, com o acústico em 8 bits).
- A mesma lógica do firmware, compilada no Mac com um shim das APIs do ESP, produz 229 quadros e rms 4203.

**Não medido:**
- Fator de tempo real no ESP32-S3. A conta (≈ 49 MMAC/s de trabalho por segundo de áudio contra um S3 a 240 MHz executando laços escalares) sugere que ele pode ficar abaixo de 1× no começo; os próximos passos (kernel int8 com o SIMD do S3 via ESP-NN, acústico e vocoder em dois núcleos, esparsificar e retreinar) estão em `esp32/pingo_bench/README.md`. Nada disso foi testado.
- Uso real de heap, o efeito do cache de dados de 64 KB e o custo do `printf` de float.
- WER desta edição (o do pingo-ptbr, 0,86% / 0,81% no MiniMax `pt`, **não** vale aqui: o vocoder e a quantização do acústico são outros).
- Qualquer teste de escuta com várias pessoas.

## Como usar

**No computador, em C puro** (sem dependências além de libm):

```bash
cd c && make && ./pingo_c ac_test_in.i32 saida.s16     # frase de teste: "Olá, eu sou a voz da SemFence."
ffmpeg -f s16le -ar 16000 -ac 1 -i saida.s16 saida.wav
make test                                               # confere o acústico contra a referência MLX
```

Para outro texto, fonemize com `c/text_to_in.py "texto" entrada.i32` (precisa do pacote `pingo/` de [semfence/pingo-ptbr](https://huggingface.co/semfence/pingo-ptbr) em `PINGO_DIR` e do `tiny_voca`/dizione-ptbr) e passe o `.i32` ao `pingo_c`.

**No ESP32-S3:** veja [`esp32/pingo_bench/README.md`](esp32/pingo_bench/README.md) (ESP-IDF 5.x/6.x ou Docker, `idf.py set-target esp32s3 && idf.py build flash monitor`).

**Reproduzir o treino do vocoder:** `training/` (`sizes.py`, `train_small.py`, exportadores `export_c.py` e `export_acoustic_c.py`). Os dados (`semfence/ptbr-synthetic-speech`) são preparados com o `fargan_demo` do Opus.
Os pesos float do xxs estão em `weights/vocoder-xxs.pth` (e `vocoder-xxs-int8sim.pth`, com o arredondamento int8 aplicado); os pesos usados no C estão em `c/xs_weights.h` (vocoder) e `c/ac_weights.h` (acústico, 4 bits).

## Limitações

- Só português do Brasil, duas vozes sintéticas no acústico original (aqui os exemplos usam a F_s4). O firmware recebe fonemas já prontos; o fonemizador está em Python, fora do chip.
- Sem saída de áudio no firmware (só mede e despeja o PCM pela serial); falta ligar o I2S/DAC.
- Áudio de 16 kHz mono com som de "chamada telefônica". Os dados de treino são fala sintética; a voz não clona ninguém.
- Modelo de pesquisa, sem garantia de qualidade para produção.

## Licença e créditos

Pesos: CC BY 4.0 (crédito: SemFence, Gustavo Toledo). Código: Apache-2.0; as partes derivadas do FARGAN mantêm a BSD-3-Clause do Xiph/Opus. Veja `NOTICE`, `THIRD_PARTY.md` e `LICENSES/`.
Autor: Gustavo Toledo · **SemFence** · https://semfence.com. English version: [`README.en.md`](README.en.md).
