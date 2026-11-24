idf_component_register(SRCS "main.c" "acoustic.c" "fargan_xs_q8.c" INCLUDE_DIRS ".")
# int8 activations + fast tanh/sigmoid; sentences up to 256 tokens (64 without PSRAM, checked in main.c; encoder RAM is proportional to the sentence)
target_compile_definitions(${COMPONENT_LIB} PRIVATE XS_FAST_MATH AC_MAXTOK=256)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -ffast-math -fno-math-errno)
