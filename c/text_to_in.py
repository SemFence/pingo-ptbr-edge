"""Fonemiza um texto para o formato de entrada do pingo em C (ac_test_in.i32).  python text_to_in.py "texto" saida.i32 [voz]"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.environ.get("PINGO_DIR", os.path.join(HERE, "..", "pingo")))   # PINGO_DIR: pasta pingo/ do repositório semfence/pingo-ptbr (speak.py) com o pacote tiny_voca instalado
import speak as sp
from tiny_voca.text import PHONE_IDS, phonemize
ph = sp.joined(phonemize(sp.spoken(sys.argv[1]))); ids, st = list(ph.ids), list(ph.stress); voice = int(sys.argv[3]) if len(sys.argv) > 3 else 0
mask = np.array([i in PHONE_IDS for i in ids]).astype(int).tolist()
np.array([len(ids), voice] + ids + st + mask, dtype=np.int32).tofile(sys.argv[2]); print(len(ids), "tokens")
