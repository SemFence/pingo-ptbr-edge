// Node helper: loads the bundled data files. In the browser use Dizione.fromUrls() instead.
import { readFileSync } from "node:fs";
import { Dizione } from "./engine.js";
export function loadDizione(dataDir = new URL("../data/", import.meta.url)) {
    const base = typeof dataDir === "string" ? new URL(`file://${dataDir.replace(/\/?$/, "/")}`) : dataDir;
    return new Dizione(readFileSync(new URL("lexicon.tsv", base), "utf8"), JSON.parse(readFileSync(new URL("open_table.json", base), "utf8")));
}
