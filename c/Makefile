# Plain-C demo on a computer (no ESP-IDF): phonemes -> 4-bit acoustic model -> int8 xxs vocoder -> 16 kHz s16 PCM.
#   make && ./pingo_c ac_test_in.i32 out.s16 && ffmpeg -f s16le -ar 16000 -ac 1 -i out.s16 out.wav
CFLAGS ?= -O2 -DXS_FAST_MATH
pingo_c: pingo_main.c acoustic.c fargan_xs_q8.c
	$(CC) $(CFLAGS) -I. -o $@ $^ -lm
ac_test: ac_test_main.c acoustic.c
	$(CC) $(CFLAGS) -I. -o $@ $^ -lm
test: ac_test
	./ac_test ac_test_in.i32 ac_test_ref.f32
clean:
	rm -f pingo_c ac_test *.s16

# WebAssembly build for web/public/assets/pingo_edge.wasm (needs emscripten: brew install emscripten)
wasm:
	emcc -O3 -DXS_FAST_MATH -ffast-math -I. --no-entry -sSTANDALONE_WASM -sINITIAL_MEMORY=16777216 -sALLOW_MEMORY_GROWTH=0 -sSTACK_SIZE=262144 \
	  -Wl,--export=pingo_tok,--export=pingo_stress,--export=pingo_mask,--export=pingo_pcm,--export=pingo_maxtok,--export=pingo_speak \
	  -o ../web/public/assets/pingo_edge.wasm pingo_wasm.c acoustic.c fargan_xs_q8.c
