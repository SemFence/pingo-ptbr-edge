// pingo-ptbr edge em JavaScript (navegador e Node): fonemizador (@semfence/dizione-ptbr) + o runtime em C compilado para WebAssembly.
// Sem dependências. Uso:
//   const pingo = await PingoEdge.create({ wasm: bytesDoWasm, dizione: new Dizione(lexiconTsv, openTable) });
//   const { audio, sampleRate } = pingo.speak("Olá, eu sou a voz da SemFence.", { voice: 0, speed: 1 });   // Float32Array, mono, 16 kHz
//   for (const s of pingo.stream(texto)) tocar(s.audio);     // frase por frase, quando o texto tem várias
import * as symbols from "./assets/dizione/symbols.js";
import { makeTokenizer } from "./tokens.js";

const symbols_ = symbols;
class TooLong extends Error {}
const PEAK = 0.99;

const ABBREVIATIONS = /(?:(?<![A-Za-zÀ-ÖØ-öø-ÿ0-9_])(?:dr|dra|prof|profa|sr|sra|srta|av|tel|obs|etc|ex|nº|vs)|(?<![A-Za-zÀ-ÖØ-öø-ÿ0-9_])[A-ZÀ-ÖØ-Ý])\.$/i;

// Frases, sem quebrar depois de títulos, iniciais ou dentro de números (igual a tiny_voca.infer.sentences).
function sentences(text) {
  const parts = [];
  let current = "";
  for (const piece of text.replace(/\s+/g, " ").trim().split(/(?<=[.!?…]["')\]]?)\s+(?=["'(\[]?[A-Z0-9À-ÖØ-Ý])/)) {
    current = current ? current + " " + piece : piece;
    if (!ABBREVIATIONS.test(current)) { parts.push(current); current = ""; }
  }
  if (current) parts.push(current);
  return parts.filter((part) => /[\p{L}\p{N}]/u.test(part));
}

// Um ponto final é acrescentado quando o texto termina sem pontuação (igual a tiny_voca.infer.finished).
function finished(text) {
  const body = text.replace(/\s+$/, "");
  if (!body) return text;
  const last = body.split(/\s+/).pop();
  const marked = /[\p{L}\p{N}]/u.test(last) ? /[.!?…,;:]["'”’)\]»]*$/.test(last) : /[.!?…]/.test(last) || /^[,;:]$/.test(last);
  return marked ? text : body + ".";
}

// Termos em inglês reescritos como se falam, antes do fonemizador (igual a pingo/pronunciation.py).
const RESPELLING = [
  [/\bHugging ?Face\b/g, "Ráguin Feice"], [/\bGitHub\b/g, "Guit rub"], [/\bstartups?\b/gi, "istártape"],
  [/\bdeploy\b/gi, "diplói"], [/\bMVP\b/g, "ême vê pê"], [/\bAPI\b/g, "a pê i"], [/\bGPU\b/g, "gê pê ú"],
  [/\bpitch\b/gi, "pítxi"], [/\bfeedback\b/gi, "fídbéque"], [/\bdatasets?\b/gi, "dêitasséti"],
  [/\bvocoder\b/gi, "vôcôder"], [/\bdecoder\b/gi, "decôder"], [/\bencoder\b/gi, "encôder"],
  [/\bONNX\b/g, "ôniks"], [/\bMLX\b/g, "ême éle xis"], [/\bWebAssembly\b/gi, "uébassémbli"],
  [/\bbrowser\b/gi, "bráuzer"], [/\bcloud\b/gi, "cláude"], [/\bsoftware\b/gi, "sófiuér"],
];
const spoken = (text) => RESPELLING.reduce((t, [pattern, said]) => t.replace(pattern, said), text);

// O fonemizador põe uma pausa entre as duas palavras de "Guit rub"; GitHub se fala numa palavra só e a vogal final é ɐ (igual a pronunciation.joined).
const JOIN = ["ɡ", "i", "t", "<sp>", "x", "u", "b"];
function joined(result) {
  const drop = new Set();
  for (let i = 0; i + JOIN.length <= result.symbols.length; i++) {
    if (JOIN.every((s, k) => result.symbols[i + k] === s)) { drop.add(i + 3); i += JOIN.length - 1; }
  }
  if (!drop.size) return result;
  const schwa = symbols_.SYMBOLS.indexOf("ɐ");
  const sym = result.symbols.map((s, k) => (drop.has(k - 2) ? "ɐ" : s)), id = result.ids.map((x, k) => (drop.has(k - 2) ? schwa : x));
  const keep = sym.map((_, k) => k).filter((k) => !drop.has(k));
  const symbols = keep.map((k) => sym[k]), stress = keep.map((k) => result.stress[k]);
  return { symbols, ids: keep.map((k) => id[k]), stress,
           text: symbols.map((s, i) => (stress[i] ? "ˈ" : "") + s).join(" ") };
}


export class PingoEdge {
  /** wasm: bytes (Uint8Array/ArrayBuffer) do pingo_edge.wasm; dizione: instância de Dizione já carregada. */
  static async create({ wasm, dizione }) {
    const instance = (await WebAssembly.instantiate(wasm, {})).instance.exports;
    return new PingoEdge(instance, dizione);
  }

  constructor(wasm, dizione) {
    this.wasm = wasm; this.dizione = dizione;
    if (wasm._initialize) wasm._initialize();
    this.tokenize = makeTokenizer(symbols);
    const index = new Map(symbols.SYMBOLS.map((s, i) => [s, i]));
    this.phoneIds = new Set([...symbols.PHONES, "<unk>"].map((s) => index.get(s)));   // fonemas e <unk>: únicos tokens com pelo menos 1 quadro
    this.config = { max_tokens: wasm.pingo_maxtok(), sample_rate: 16000, voices: ["F_s4", "M_sm4"] };
    this.synthesizeTokens(this.tokenize(dizione.pieces("Olá.")), 0, 1);               // aquece e confere
  }

  get sampleRate() { return 16000; }

  synthesizeTokens(phones, voice, speed) {
    const wasm = this.wasm, n = phones.ids.length;
    if (n > wasm.pingo_maxtok()) throw new TooLong();
    const put = (ptr, values) => new Int32Array(wasm.memory.buffer, ptr, n).set(values);
    put(wasm.pingo_tok(), phones.ids); put(wasm.pingo_stress(), phones.stress);
    put(wasm.pingo_mask(), phones.ids.map((id) => (this.phoneIds.has(id) ? 1 : 0)));
    const samples = wasm.pingo_speak(n, voice, speed);
    if (samples < 0) throw new TooLong();
    return new Float32Array(wasm.memory.buffer, wasm.pingo_pcm(), samples).slice();   // cópia: o buffer do wasm é reaproveitado
  }

  // Uma fala tem no máximo max_tokens fonemas e 30 s; frases maiores são cortadas em vírgulas e depois em palavras.
  *pieces(sentence, voice, speed) {
    const raw = this.tokenize(this.dizione.pieces(sentence));
    const result = raw && joined(raw);
    if (!result) return;
    if (result.ids.length <= this.config.max_tokens - 12) {
      let audio = null;
      try { audio = this.synthesizeTokens(result, voice, speed); } catch (error) { if (!(error instanceof TooLong)) throw error; }
      if (audio) { yield { phonemes: result.text, audio }; return; }
    }
    const parts = sentence.split(/(?<=[,;:—])\s+/);
    if (parts.length > 1) { for (const part of parts) yield* this.pieces(part, voice, speed); return; }
    const words = sentence.split(" ");
    if (words.length < 2) return;
    const half = Math.ceil(words.length / 2);
    yield* this.pieces(words.slice(0, half).join(" "), voice, speed);
    yield* this.pieces(words.slice(half).join(" "), voice, speed);
  }

  /** Gera o áudio frase por frase: { index, total, phonemes, audio } com audio já limitado a 0,99. */
  *stream(text, { voice = 0, speed = 1, respelling = true } = {}) {
    const list = sentences(finished(respelling ? spoken(text) : text));
    let index = 0;
    for (const sentence of list) for (const piece of this.pieces(sentence, voice, speed)) { limitPeak(piece.audio); yield { index: index++, total: list.length, ...piece }; }
  }

  /** Texto -> { audio: Float32Array (mono, 16 kHz), sampleRate, phonemes: [..] }. Lança erro se não há nada em português para ler. */
  speak(text, options = {}) {
    const chunks = [], phonemes = [];
    for (const s of this.stream(text, options)) { chunks.push(s.audio); phonemes.push(s.phonemes); }
    if (!chunks.length) throw new Error("O pingo-ptbr só lê português do Brasil (pt-BR) por enquanto.");
    const audio = new Float32Array(chunks.reduce((sum, c) => sum + c.length, 0));
    let offset = 0;
    for (const c of chunks) { audio.set(c, offset); offset += c.length; }
    return { audio, sampleRate: 16000, phonemes };
  }
}

// O vocoder às vezes passa de 1,0, o que cortaria no arquivo de 16 bits: reduz o trecho todo nesse caso.
function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
}

/** Float32 mono -> arquivo WAV de 16 bits (Uint8Array). */
export function wavBytes(samples, sampleRate = 16000) {
  const out = new Uint8Array(44 + samples.length * 2), v = new DataView(out.buffer);
  const tag = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  tag(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); tag(8, "WAVEfmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  tag(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), true);
  return out;
}
