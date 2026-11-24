"""Phonemizes a text into the input format of the C pingo (ac_test_in.i32).  python text_to_in.py "text" out.i32 [voice]"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.environ.get("PINGO_DIR", os.path.join(HERE, "..", "pingo")))   # PINGO_DIR: the pingo/ folder of semfence/pingo-ptbr (speak.py), with the tiny_voca package installed
import speak as sp
from tiny_voca.text import PHONE_IDS, phonemize
ph = sp.joined(phonemize(sp.spoken(sys.argv[1]))); ids, st = list(ph.ids), list(ph.stress); voice = int(sys.argv[3]) if len(sys.argv) > 3 else 0
mask = np.array([i in PHONE_IDS for i in ids]).astype(int).tolist()
np.array([len(ids), voice] + ids + st + mask, dtype=np.int32).tofile(sys.argv[2]); print(len(ids), "tokens")
