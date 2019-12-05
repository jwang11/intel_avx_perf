#include <x86intrin.h>
#include "perf.h"

#define col 1024 * 2
#define row  64

int mat[row][col];
int mutiplier[col];
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
      mat[i][j] = (int)(rand() % (1024*32));
    }   
  }
  for (int j = 0; j < col; j++) {
    mutiplier[j] = (int)(rand() % (1024*32));
  }
}

void calc_non(void) {
  for(int j = 0; j < row; j++) {
    int sum = 0;
    int *vec = mat[j];

    for(int i = 0; i < col; i++)
      sum += vec[i] * mutiplier[i];

    result[j] = sum;
  }
}

void calc_avx2(void) {
  #define VLEN (256/8/sizeof(int))
  const int col_aligned = col - col % VLEN;
  int *dst_arr;

  __m256i op0, op1, tmp_vec;
  volatile __m256i  tgt;

  for (int i = 0; i < row; i++) {
    int sum = 0;
    tgt = _mm256_setzero_si256();
    for (int j = 0; j < col_aligned; j += VLEN) {
      op0 = _mm256_load_si256((__m256i *)&mutiplier[j]);
      op1 = _mm256_load_si256((__m256i *)&mat[i][j]);
      tmp_vec = _mm256_mullo_epi32(op0, op1);
      tgt = _mm256_add_epi32(tmp_vec, tgt);
    }

    dst_arr = (int *)&tgt;
    for (int k = 0; k < VLEN; k++)
      sum += dst_arr[k];

    for (int l = col_aligned; l < col; l++)
      sum += mat[i][l] * mutiplier[l];

    result[i] = sum;
  }
  #undef VLEN
}

void calc_avx512(void) {
  #define VLEN (512/8/sizeof(int))
  const int col_aligned = col - col % VLEN;
  int *dst_arr;

  __m512i op0, op1, tmp_vec;
  volatile __m512i  tgt;

  for (int i = 0; i < row; i++) {
    int sum = 0;

    tgt = _mm512_setzero_si512();
    for (int j = 0; j < col_aligned; j += VLEN) {
      op0 = _mm512_load_si512(&mutiplier[j]);
      op1 = _mm512_load_si512(&mat[i][j]);
      tmp_vec = _mm512_mullo_epi32(op0, op1);
      tgt = _mm512_add_epi32(tmp_vec, tgt);
    }

    dst_arr = (int *)&tgt;
    for (int k = 0; k < VLEN; k++)
      sum += dst_arr[k];

    for (int l = col_aligned; l < col; l++)
      sum += mat[i][l] * mutiplier[l];

    result[i] = sum;
  }
  #undef VLEN
}

int main() {
  init();
  TIME(LOOP(calc_non()));
  DEBUG
  TIME(LOOP(calc_avx2()));
  DEBUG
  TIME(LOOP(calc_avx512()));
  DEBUG
}
