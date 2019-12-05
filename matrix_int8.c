#include <x86intrin.h>
#include "perf.h"

#define col 1024 * 4
#define row  64

int8_t mat[row][col];
int8_t mutiplier[col];
int16_t op4_int16[32];
int result[row];

#define DEBUG { \
  for (int i = 0; i < 8; i++) \
    printf("%d, ", result[i]); \
  printf("...\n\n"); \
}

void init(void) {
  srand(time(0));
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      mat[i][j] = (int8_t)(rand() % 32);
    }   
  }
  for (int j = 0; j < col; j++) {
    mutiplier[j] = (int8_t)(rand() % 32);
  }

  for (int i = 0; i < 32; i++) {
       op4_int16[i] = 1;
  }
}

void calc_non(void) {
  for(int j = 0; j < row; j++) {
    int sum = 0;
    int8_t *vec = mat[j];

    for(int i = 0; i < col; i += 4) {
      int16_t hw_0 = (int16_t)(vec[i] * mutiplier[i] + vec[i + 1] * mutiplier[i + 1]);
      int16_t hw_1 = (int16_t)(vec[i + 2] * mutiplier[i + 2] + vec[i + 3] * mutiplier[i + 3]);
      sum += (int)hw_0 + (int)hw_1;
    }
    result[j] = sum;
  }
}

void calc_avx512(void) {
  #define VLEN (512/8/sizeof(int))
  const int col_aligned = col - col % VLEN;

  __m512i op0, op1;
  volatile __m512i tgt;

  __m512i v4_int16 = _mm512_load_si512(&op4_int16);

  for (int i = 0; i < row; i++) {
    int sum = 0;

    tgt = _mm512_setzero_si512();
    for (int j = 0; j < col_aligned; j += 64) {
      op0 = _mm512_load_si512(&mutiplier[j]);
      op1 = _mm512_load_si512(&mat[i][j]);
      __m512i vresult1 = _mm512_maddubs_epi16(op0, op1);
      __m512i vresult2 = _mm512_madd_epi16(vresult1, v4_int16);
      tgt = _mm512_add_epi32(vresult2, tgt);
    }

    int *presult = (int *)&tgt;
    for (int k = 0; k < VLEN; k++)
      sum += presult[k];

    for (int l = col_aligned; l < col; l++)
      sum += (int)(mat[i][l] * mutiplier[l]);

    result[i] = sum;
  }
  #undef VLEN
}

void calc_avx512_vnni(void) {
  #define VLEN (512/8/sizeof(int))
  const int col_aligned = col - col % VLEN;

  __m512i op0, op1;
  volatile __m512i tgt;

  for (int i = 0; i < row; i++) {
    int sum = 0;

    tgt = _mm512_setzero_si512();
    for (int j = 0; j < col_aligned; j += 64) {
      op0 = _mm512_load_si512(&mutiplier[j]);
      op1 = _mm512_load_si512(&mat[i][j]);
      tgt = _mm512_dpbusds_epi32(tgt, op0, op1);
    }
    int *presult = (int *)&tgt;
    for (int k = 0; k < VLEN; k++)
      sum += presult[k];

    for (int l = col_aligned; l < col; l++)
      sum += (int)(mat[i][l] * mutiplier[l]);

    result[i] = sum;
  }
  #undef VLEN
}

int main() {
  init();
  TIME(LOOP(calc_non()));
  DEBUG
  TIME(LOOP(calc_avx512_vnni()));
  DEBUG
  TIME(LOOP(calc_avx512()));
  DEBUG
}
