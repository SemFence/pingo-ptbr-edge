# Third-party materials

The weights of pingo-ptbr-edge are CC BY 4.0 (`LICENSE`); the code is Apache-2.0 (the FARGAN-derived parts keep BSD-3-Clause).

## FARGAN (Xiph/Opus)

`c/fargan_xs*.c`, `training/fargan_small.py`, `training/stft_loss.py`, `training/dataset.py` and `training/filters.py` follow or derive from the
FARGAN vocoder of Xiph/Opus (BSD-3-Clause), source <https://github.com/xiph/opus/tree/main/dnn/torch/fargan>, with narrower layers
(see `training/sizes.py`) and an int8 C runtime. Retain `LICENSES/Opus-BSD-3-Clause.txt` and the upstream patent notices when redistributing.
PiVo-1.4M (Empero Research, Apache-2.0, <https://huggingface.co/empero-ai/PiVo-1.4M>) converted FARGAN to PyTorch; its license and notice are in
`LICENSES/PiVo-Apache-2.0.txt` and `LICENSES/PiVo-NOTICE.txt`. The reduced vocoders here were trained from random initialization; no PiVo weights are used.

## Espressif ESP-IDF

`esp32/pingo_bench` is an ESP-IDF project (Apache-2.0, <https://github.com/espressif/esp-idf>). ESP-IDF is not redistributed here.

## Data and tools (not distributed, not used at inference)

- Training speech: `semfence/ptbr-synthetic-speech`, generated with Chatterbox Multilingual pt-BR (Resemble AI, MIT), CC BY 4.0. Texts: `semfence/ptbr-sentences` (includes Mozilla Common Voice sentences, CC0).
- Acoustic features for the vocoder training come from `fargan_demo` (Opus, BSD-3-Clause), not distributed here.
- Phonemizer for new sentences: dizione-ptbr (CC BY 4.0 lexicon derived from the Montreal Forced Aligner Brazilian Portuguese dictionary).

## PyTorch/MLX code and the browser demo

- `python/pingo_edge/acoustic_mlx.py` and `pronunciation.py` are copies of `pingo/pingo_mlx.py` (acoustic part) and `pingo/pronunciation.py` from semfence/pingo-ptbr; `_torch_acoustic_model.py` is `training/model.py` from the same repository (code Apache-2.0; design follows PiVo-1.4M, Apache-2.0).
- `web/public/assets/dizione/` is the compiled dizione-ptbr-js phonemizer (Apache-2.0 code; lexicon CC BY 4.0, derived from the Montreal Forced Aligner Brazilian Portuguese dictionary, see `assets/dizione/data/LICENSE.mfa-dictionary.md`). `web/public/{player,tokens}.js` and the structure of `worker.js`/`index.html` come from the pingo-ptbr demo. `web/public/pingo.js` factors the demo worker logic into a browser/Node library.
