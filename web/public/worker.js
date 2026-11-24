// pingo-ptbr edge no navegador, fora da thread principal: usa a biblioteca pingo.js (fonemizador + runtime em C em WebAssembly).
// Mensagens de entrada: {type: "load"} e {type: "speak", id, text, voice, speed}.
// Saída: status, ready, sentence (com o áudio assim que fica pronto), done (clipe inteiro), error.
import { Dizione } from "./assets/dizione/engine.js";
import { PingoEdge } from "./pingo.js";

let pingo;
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

async function load() {
  status("Carregando o fonemizador");
  const [lexicon, table] = await Promise.all([
    download("assets/dizione/data/lexicon.tsv", "Carregando o dicionário").then((b) => new TextDecoder().decode(b)),
    download("assets/dizione/data/open_table.json", "Carregando o dicionário").then((b) => JSON.parse(new TextDecoder().decode(b))),
  ]);
  const wasm = await download("assets/pingo_edge.wasm", "Carregando o modelo");
  status("Iniciando o modelo");
  pingo = await PingoEdge.create({ wasm, dizione: new Dizione(lexicon, table) });
}

async function speak({ id, text, voice, speed }) {
  const started = performance.now();
  const chunks = [];
  let total = 0;
  for (const s of pingo.stream(text, { voice, speed })) {
    chunks.push(s.audio); total = s.total;
    const heard = s.audio.slice();   // a página toca esta frase já, antes de saber o clipe todo
    postMessage({ type: "sentence", id, index: s.index, total, phonemes: s.phonemes, audio: heard, sampleRate: 16000 }, [heard.buffer]);
  }
  if (!chunks.length) { postMessage({ type: "error", id, message: "O pingo-ptbr só lê português do Brasil (pt-BR) por enquanto.", plain: true }); return; }
  const audio = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const c of chunks) { audio.set(c, offset); offset += c.length; }
  const elapsed = performance.now() - started;
  postMessage({ type: "done", id, audio, sampleRate: 16000, phonemeTime: 0, modelTime: elapsed, provider: "wasm (C)", totalTime: elapsed }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") { await load(); postMessage({ type: "ready", voices: ["F_s4", "M_sm4"], sampleRate: 16000, provider: "wasm" }); }
    else if (data.type === "speak") await speak(data);
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String(error && error.message || error) });
  }
};
