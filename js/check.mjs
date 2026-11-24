// Checks the JS version against the C code: the same .wasm and the same test sentence (c/ac_test_in.i32) must give 229 frames and rms 4203;
// and the text "Olá, eu sou a voz da SemFence." phonemized in JS must give the same input as Python (30 tokens).   node js/check.mjs
import { readFileSync } from "node:fs";
import { loadDizione } from "../web/public/assets/dizione/node.js";
import { PingoEdge } from "../web/public/pingo.js";
const here = new URL("../web/public/", import.meta.url);
const pingo = await PingoEdge.create({ wasm: readFileSync(new URL("assets/pingo_edge.wasm", here)), dizione: loadDizione(new URL("assets/dizione/data/", here)) });
const raw = readFileSync(new URL("../c/ac_test_in.i32", import.meta.url)), a = new Int32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)), L = a[0];
const w = pingo.wasm, put = (p, v) => new Int32Array(w.memory.buffer, p, L).set(v);
put(w.pingo_tok(), a.subarray(2, 2 + L)); put(w.pingo_stress(), a.subarray(2 + L, 2 + 2 * L)); put(w.pingo_mask(), a.subarray(2 + 2 * L, 2 + 3 * L));
const n = w.pingo_speak(L, a[1], 1), pcm = new Float32Array(w.memory.buffer, w.pingo_pcm(), n);
let ss = 0; for (const v of pcm) ss += (v * 32768) ** 2;
const rms = Math.sqrt(ss / n), frames = n / 160;
const tokens = pingo.tokenize(pingo.dizione.pieces("Olá, eu sou a voz da SemFence.")).ids.length;
console.log(`quadros ${frames} (esperado 229) | rms ${rms.toFixed(0)} (esperado 4203) | tokens do texto fonemizado em JS ${tokens} (esperado 30)`);
if (frames !== 229 || Math.abs(rms - 4203) > 5 || tokens !== 30) { console.error("FALHOU"); process.exit(1); }
console.log("ok");
