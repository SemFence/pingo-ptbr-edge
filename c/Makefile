# Demo em C puro no computador (sem ESP-IDF): fonemas -> acústico 4 bits -> vocoder xxs int8 -> PCM s16 16 kHz.
#   make && ./pingo_c ac_test_in.i32 saida.s16 && ffmpeg -f s16le -ar 16000 -ac 1 -i saida.s16 saida.wav
CFLAGS ?= -O2 -DXS_FAST_MATH
pingo_c: pingo_main.c acoustic.c fargan_xs_q8.c
	$(CC) $(CFLAGS) -I. -o $@ $^ -lm
ac_test: ac_test_main.c acoustic.c
	$(CC) $(CFLAGS) -I. -o $@ $^ -lm
test: ac_test
	./ac_test ac_test_in.i32 ac_test_ref.f32
clean:
	rm -f pingo_c ac_test *.s16
