// pingo-ptbr edge no navegador: fonemizador (@semfence/dizione-ptbr, TypeScript) + o MESMO runtime em C da placa (acústico 4 bits
// + vocoder xxs int8) compilado para WebAssembly (assets/pingo_edge.wasm, 308 KB, sem ONNX Runtime), fora da thread principal.
// Mensagens de entrada: {type: "load"} e {type: "speak", id, text, voice, speed}.
// Saída: status, ready, sentence (com o áudio assim que fica pronto), done (clipe inteiro), error.
import { Dizione } from "./assets/dizione/engine.js";
import * as symbols from "./assets/dizione/symbols.js";
const symbols_ = symbols;
import { makeTokenizer } from "./tokens.js";

let dizione, tokenize, config, wasm, phoneIds;
const status = (text, progress) => postMessage({ type: "status", text, progress });

// Suba quando os arquivos de assets/ mudarem, para os navegadores baixarem de novo.
const VERSION = "1";

// Arquivos baixados ficam no cache do navegador: a segunda visita começa sem rede.
async function download(url, label) {
  const key = `${url}?v=${VERSION}`;
  let store = null;
  try {
    store = await caches.open("pingo-ptbr-" + VERSION);
    const hit = await store.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (error) { store = null; }
  const bytes = await fetchBytes(key, label);
  if (store) store.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

async function fetchBytes(url, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(label, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

async function loadPhonemizer() {
  status("Carregando o fonemizador");
  const [lexicon, table] = await Promise.all([
    download("assets/dizione/data/lexicon.tsv", "Carregando o dicionário").then((b) => new TextDecoder().decode(b)),
    download("assets/dizione/data/open_table.json", "Carregando o dicionário").then((b) => JSON.parse(new TextDecoder().decode(b))),
  ]);
  dizione = new Dizione(lexicon, table);
  tokenize = makeTokenizer(symbols);
  // Fonemas e <unk> são os únicos tokens com pelo menos 1 quadro (como no treino).
  const index = new Map(symbols.SYMBOLS.map((s, i) => [s, i]));
  phoneIds = new Set([...symbols.PHONES, "<unk>"].map((s) => index.get(s)));
}

async function loadModel() {
  config = await (await fetch("assets/model.json")).json();
  const bytes = await download("assets/pingo_edge.wasm", "Carregando o modelo");
  status("Iniciando o modelo");
  wasm = (await WebAssembly.instantiate(bytes, {})).instance.exports;
  if (wasm._initialize) wasm._initialize();
  config.max_tokens = wasm.pingo_maxtok();
  await synthesize(tokenize(dizione.pieces("Olá.")), 0, 1);   // aquece e confere que o navegador roda tudo
}

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

class TooLong extends Error {}

// Uma fala tem no máximo max_tokens fonemas e 30 s; frases maiores são cortadas em vírgulas e depois em palavras.
async function pieces(sentence, voice, speed, emit) {
  const raw = tokenize(dizione.pieces(sentence));
  const result = raw && joined(raw);
  if (!result) return;
  if (result.ids.length <= config.max_tokens - 12) {
    try { emit(result, synthesize(result, voice, speed)); return; }
    catch (error) { if (!(error instanceof TooLong)) throw error; }
  }
  const parts = sentence.split(/(?<=[,;:—])\s+/);
  if (parts.length > 1) { for (const part of parts) await pieces(part, voice, speed, emit); return; }
  const words = sentence.split(" ");
  if (words.length < 2) return;
  const half = Math.ceil(words.length / 2);
  await pieces(words.slice(0, half).join(" "), voice, speed, emit);
  await pieces(words.slice(half).join(" "), voice, speed, emit);
}

const PEAK_FRAMES = 3000;
function synthesize(phones, voice, speed) {
  const n = phones.ids.length;
  if (n > wasm.pingo_maxtok()) throw new TooLong();
  const put = (ptr, values) => new Int32Array(wasm.memory.buffer, ptr, n).set(values);
  put(wasm.pingo_tok(), phones.ids); put(wasm.pingo_stress(), phones.stress);
  put(wasm.pingo_mask(), phones.ids.map((id) => (phoneIds.has(id) ? 1 : 0)));
  const samples = wasm.pingo_speak(n, voice, speed);
  if (samples < 0) throw new TooLong();
  return new Float32Array(wasm.memory.buffer, wasm.pingo_pcm(), samples).slice();   // cópia: o buffer do wasm é reaproveitado
}

// O vocoder às vezes passa de 1,0, o que cortaria no arquivo de 16 bits: reduz o clipe todo nesse caso.
const PEAK = 0.99;
function limitPeak(samples) {
  let loudest = 0;
  for (let i = 0; i < samples.length; i++) loudest = Math.max(loudest, Math.abs(samples[i]));
  if (loudest > PEAK) for (let i = 0; i < samples.length; i++) samples[i] *= PEAK / loudest;
}

const PORTUGUESE_ONLY = "O pingo-ptbr só lê português do Brasil (pt-BR) por enquanto.";

async function speak({ id, text, voice, speed }) {
  const started = performance.now();
  const list = sentences(finished(spoken(text)));
  const chunks = [];
  let modelTime = 0, phonemeTime = 0;
  for (let index = 0; index < list.length; index++) {
    let mark = performance.now();
    const emit = (phones, audio) => {
      modelTime += performance.now() - mark;
      chunks.push(audio);
      const heard = audio.slice();   // a página toca esta frase já, antes de saber o clipe todo
      limitPeak(heard);
      postMessage({ type: "sentence", id, index, total: list.length, phonemes: phones.text, audio: heard,
                    sampleRate: config.sample_rate }, [heard.buffer]);
      mark = performance.now();
    };
    await pieces(list[index], voice, speed, emit);
  }
  if (!chunks.length) {
    postMessage({ type: "error", id, message: PORTUGUESE_ONLY, plain: true });
    return;
  }
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  limitPeak(audio);
  postMessage({ type: "done", id, audio, sampleRate: config.sample_rate, phonemeTime, modelTime, provider: "wasm (C)",
                totalTime: performance.now() - started }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await loadPhonemizer();
      await loadModel();
      postMessage({ type: "ready", voices: config.voices, sampleRate: config.sample_rate, provider: "wasm (C)" });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error) });
  }
};
