idf_component_register(SRCS "main.c" "acoustic.c" "fargan_xs_q8.c" INCLUDE_DIRS ".")
# ativações int8 + tanh/sigmoid rápidos; frases de até 256 tokens (64 sem PSRAM, checado em main.c; a RAM do encoder é proporcional à frase)
target_compile_definitions(${COMPONENT_LIB} PRIVATE XS_FAST_MATH AC_MAXTOK=256)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -ffast-math -fno-math-errno)
