#!/usr/bin/env node
// pingo-ptbr edge in Node (no dependencies): node js/cli.mjs "Olá, eu sou a voz da SemFence." -o out.wav [--voice M_sm4] [--speed 1.1]
import { readFileSync, writeFileSync } from "node:fs";
import { Dizione } from "../web/public/assets/dizione/engine.js";
import { loadDizione } from "../web/public/assets/dizione/node.js";
import { PingoEdge, wavBytes } from "../web/public/pingo.js";

const args = process.argv.slice(2);
const opt = (name, fallback) => { const i = args.indexOf(name); if (i < 0) return fallback; const [, v] = args.splice(i, 2); return v; };
const output = opt("-o", "out.wav"), voiceName = opt("--voice", "F_s4"), speed = Number(opt("--speed", "1"));
const text = args.join(" ");
if (!text) { console.error('uso: node js/cli.mjs "texto" [-o saida.wav] [--voice F_s4|M_sm4] [--speed 1.0]'); process.exit(1); }
const voice = { F_s4: 0, M_sm4: 1 }[voiceName];
if (voice === undefined) { console.error("voz: F_s4 ou M_sm4"); process.exit(1); }

const t0 = performance.now();
const here = new URL("../web/public/", import.meta.url);
const pingo = await PingoEdge.create({ wasm: readFileSync(new URL("assets/pingo_edge.wasm", here)), dizione: loadDizione(new URL("assets/dizione/data/", here)) });
const t1 = performance.now();
const { audio, sampleRate } = pingo.speak(text, { voice, speed });
const t2 = performance.now();
writeFileSync(output, wavBytes(audio, sampleRate));
const secs = audio.length / sampleRate;
console.log(`${output}: ${secs.toFixed(2)} s de áudio | carga ${((t1 - t0) / 1000).toFixed(2)} s | síntese ${((t2 - t1) / 1000).toFixed(2)} s (${(secs / ((t2 - t1) / 1000)).toFixed(1)}x tempo real) | node ${process.version}`);
