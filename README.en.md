# pingo-ptbr-edge

Modelo e pesos também em / model and weights also on Hugging Face: <https://huggingface.co/semfence/pingo-ptbr-edge>

The microcontroller edition of [pingo-ptbr](https://huggingface.co/semfence/pingo-ptbr), a Brazilian Portuguese (pt-BR) text-to-speech model: **379,814 parameters** in total, in plain C (libm only), no PyTorch, no network.
The acoustic model (270,081 parameters) stores its weights in **4 bits**; the vocoder is a reduced FARGAN, **xxs, with 109,733 parameters** (int8, 107 KiB), trained from scratch.
The original pingo has 1,086,678 parameters, 817 k of them the vocoder. Here the vocoder is 7.4× smaller and costs **~39.6 MMAC/s instead of 299**.

> **Honest status.** The firmware **builds for esp32s3** and the same code runs on a Mac with output matching the reference, but it has **never been run on a board**.
> There is no real-time measurement on the ESP32-S3, no measured RAM use and no WER for this edition. What is measured and what is an estimate is separated below.
> If you run it on a board, what `esp32/pingo_bench` prints over serial is exactly the missing data.

## Listen

Samples generated **by the C code** (`c/`), unedited, voice F_s4 (Portuguese): `samples/ola_semfence_edge.wav` ("Olá, eu sou a voz da SemFence.") and `samples/startup_edge.wav` ("Sistema iniciado. Pingo pronto para falar.").
`samples/vocoder_xs_vs_xxs/` has the same sentences with the xs and xxs vocoders (int8); `samples/acoustic_bits/` has the acoustic model in float, 8, 6, 5 and 4 bits.
Listening was done by the author alone, it is not a listening test. The sound is pingo's: "phone call" quality, slightly metallic.

## What changed from pingo-ptbr

| | pingo-ptbr | pingo-ptbr-edge |
|---|---:|---:|
| Acoustic | 270,081 params, 5 to 8 bits | 270,081 params, **4 bits** (131 KiB) |
| Vocoder | FARGAN 816,597 params (from PiVo) | **xxs 109,733 params**, int8 (107 KiB), trained from scratch |
| Vocoder cost | ~299 MMAC/s | **~39.6 MMAC/s** |
| Runtime | MLX / ONNX | **plain C** (int8 activations, rational tanh/sigmoid), **PyTorch**, **MLX**, **WebAssembly** |
| Target | Mac, browser | **ESP32-S3** (and any CPU with a C compiler) |

Intermediate sizes also trained (`training/sizes.py`): **s** (262 k params, 90 MMAC/s) and **xs** (160 k params, 56 MMAC/s).

## The xxs vocoder

Narrower FARGAN (`C=24, d1=32, d2=48, F=48, H1=48, H2=40, H3=40, S=40`). Opus recipe (multi-resolution STFT loss + 0.03 × cosine, AdamW, 8-bit soft quantization during training) on 500 k sequences of synthetic speech (twice the xs data), 45 epochs (best: 38), one rented RTX 3090 for 133 min (~US$0.5).

Validation spectral loss (`val_spect`, lower is better; same set, same features):

| | params | MMAC/s | val_spect |
|---|---:|---:|---:|
| s | 262 k | 90 | 0.2487 |
| **xxs** (float) | 110 k | 38.7 | **0.2656** |
| xxs (simulated int8) | 110 k | 38.7 | 0.2661 |
| xs | 160 k | 56 | 0.2814 |

Caveat: xxs saw **twice the data** of xs (500 k vs 250 k sequences), so beating xs does **not** show the architecture is better. Spectral loss is also not perceived quality; the choice of xxs was made by ear by the author.

## 4-bit acoustic model

Post-training, symmetric, per-row quantization (values −7..7, one float scale per row), two weights per byte; only 2-D weights are quantized, vectors stay float.
Measured side effect: **predicted durations get ~8% shorter** (229 vs 248 frames on the test sentence), so speech comes out a bit faster; the speed parameter compensates.
At 5 bits durations grow 2 to 5%, at 6 bits they shrink ~1.5%, at 8 bits ±0.6%, so the objective feature comparison is not direct and 4 bits was chosen by ear.

## Size on the ESP32-S3

Measured with `idf.py size` (ESP-IDF 6.x in Docker, esp32s3, trimmed in `sdkconfig.defaults`):

| | firmware | fixed internal RAM (of 342 KB) |
|---|---:|---:|
| no PSRAM | 443 KB | 148 KB |
| with PSRAM (`sdkconfig.psram`) | 448 KB | 151 KB |

Vocoder weights live in internal SRAM (re-read 400×/s); acoustic weights stay in flash. The encoder uses memory proportional to the sentence (23 KB for 30 tokens) and, with PSRAM, moves there and the limit grows from 64 to 256 tokens.
**Estimate (not measured on a board):** ~50 KB peak internal working RAM on the test sentence.

## What is measured and what is not

Measured (on a Mac or in the build):
- Parameters, MMAC/s, firmware and fixed-RAM sizes (above).
- The C acoustic model matches the MLX reference (same rounded weights): 0.0000% relative rms error on 30 tokens / 229 frames.
- The C vocoder (float) matches PyTorch (error 1e-6). The int8-activation version diverges sample by sample from PyTorch (the vocoder is recursive), so it was validated by rms and listening (rms 4577 vs 4604 for xs on the same sentence, with the 8-bit acoustic model).
- The firmware logic, compiled on a Mac with a shim of the ESP APIs, gives 229 frames and rms 4203.

**Not measured:**
- Real-time factor on the ESP32-S3. The arithmetic (≈ 49 MMAC per second of audio against a 240 MHz S3 running scalar loops) suggests it may start below 1×; next steps (an int8 kernel with the S3 SIMD via ESP-NN, acoustic and vocoder on two cores, sparsify and retrain) are in `esp32/pingo_bench/README.md` and untested.
- Real heap use, the effect of the 64 KB data cache, the cost of float `printf`.
- WER of this edition (pingo-ptbr's 0.86% / 0.81% on MiniMax `pt` does **not** apply: the vocoder and the acoustic quantization differ).
- Any listening test with several people.

## Usage

**On a computer, plain C** (libm only):

```bash
cd c && make && ./pingo_c ac_test_in.i32 out.s16        # test sentence
ffmpeg -f s16le -ar 16000 -ac 1 -i out.s16 out.wav
make test                                               # checks the acoustic model against the MLX reference
```

For other text, phonemize with `c/text_to_in.py "text" input.i32` (needs the `pingo/` package from [semfence/pingo-ptbr](https://huggingface.co/semfence/pingo-ptbr) in `PINGO_DIR` and `tiny_voca`/dizione-ptbr) and pass the `.i32` to `pingo_c`.

**In PyTorch and MLX** (`python/`, the SAME weights as the firmware: `weights/pingo-edge.npz`, generated from the C headers by `tools/headers_to_npz.py`):

```bash
pip install -r python/requirements.txt
cd python
python -m pingo_edge --backend torch --text "Olá, eu sou a voz da SemFence." -o out.wav
python -m pingo_edge --backend mlx   --text "Olá, eu sou a voz da SemFence." -o out.wav       # Apple Silicon
python check.py                                                                                # checks both against the C code
```

The acoustic model (4-bit, dequantized) and the xxs vocoder (int8, dequantized) run in float. Measured by `check.py`: features match the C/MLX reference (relative rms error 2.5e-7, 229 frames) and the MLX vocoder matches PyTorch (1.1e-6).
The PyTorch vocoder is within ~1.2% (relative rms) of the training model with simulated int8 weights (`weights/vocoder-xxs-int8sim.pth`, training-time quantization noise switched off); same audio rms (4205 vs 4209). The vocoder is recursive and amplifies rounding differences.
They do not run the C code's int8 activations, so the audio is not sample-identical to the firmware's (same rms, 4203 vs 4205).

**In the browser** (`web/public/`, the same C compiled to WebAssembly, 308 KB, no ONNX Runtime):

```bash
cd web/public && python3 -m http.server 8000      # open http://localhost:8000
```

The phonemizer (dizione-ptbr, TypeScript) and its lexicon are bundled; the first visit downloads ~1.3 MB. Measured in Claude's browser: 4.0 s of audio in 0.08 s (~50× real time) on a Mac CPU; that is a desktop, not the ESP32.
To rebuild the `.wasm`: `cd c && make wasm` (needs emscripten).

**On the ESP32-S3:** see [`esp32/pingo_bench/README.md`](esp32/pingo_bench/README.md) (Portuguese; `idf.py set-target esp32s3 && idf.py build flash monitor`).

**Reproduce the vocoder training:** `training/` (`sizes.py`, `train_small.py`, exporters `export_c.py` and `export_acoustic_c.py`). Data (`semfence/ptbr-synthetic-speech`) is prepared with Opus's `fargan_demo`.
Float xxs weights: `weights/vocoder-xxs.pth` (and `vocoder-xxs-int8sim.pth`, with int8 rounding applied); the weights used in C are `c/xs_weights.h` (vocoder) and `c/ac_weights.h` (acoustic, 4 bits).

## Limitations

- Brazilian Portuguese only. The firmware takes ready-made phonemes; the phonemizer is Python and runs off-chip.
- No audio output in the firmware (it only measures and dumps PCM over serial); I2S/DAC is not wired up.
- 16 kHz mono with "phone call" sound. Training data is synthetic speech; the voice clones nobody.
- Research model, no production-quality guarantee.

## License and credits

Weights: CC BY 4.0 (credit: SemFence, Gustavo Toledo). Code: Apache-2.0; FARGAN-derived parts keep Xiph/Opus's BSD-3-Clause. See `NOTICE`, `THIRD_PARTY.md`, `LICENSES/`.
Author: Gustavo Toledo · **SemFence** · https://semfence.com. Portuguese version: [`README.md`](README.md).
