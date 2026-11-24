#!/usr/bin/env python3
"""Converts the PCM printed by the benchmark (PINGO_DUMP_PCM=1) into a 16 kHz WAV.
  idf.py monitor | tee log.txt      # let it run until "fim" appears
  python3 tools/pcm_from_serial.py log.txt out.wav
PCM lines are plain hexadecimal (4 digits per sample); log lines are ignored."""
import re, sys, wave, struct
hexline = re.compile(r"^[0-9a-f]+$")
samples = []
for line in open(sys.argv[1], errors="ignore"):
    line = line.strip()
    if len(line) % 4 == 0 and len(line) >= 40 and hexline.match(line):
        for i in range(0, len(line), 4):
            v = int(line[i:i + 4], 16); samples.append(v - 65536 if v >= 32768 else v)
if not samples: sys.exit("nenhuma linha de PCM encontrada (PINGO_DUMP_PCM=1?)")
with wave.open(sys.argv[2], "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(struct.pack("<%dh" % len(samples), *samples))
print(f"{len(samples)} amostras = {len(samples)/16000:.2f} s -> {sys.argv[2]}")
