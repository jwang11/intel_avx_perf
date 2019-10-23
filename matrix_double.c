/*
Copyright (c) 2019, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of Intel Corporation nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <x86intrin.h>
#include "perf.h"

#define col 1024
#define row  64

double mat[row][col];
double mutiplier[col];
double result[row];

#define DEBUG { \
  for (int i = 0; i < 8; i++) \
    printf("%.4f, ", result[i]); \
  printf("...\n\n"); \
}

void init(void) {
  srand(time(0));
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      mat[i][j] = (double)(rand() % 1000) / 300.0f;
    }   
  }
  for (int j = 0; j < col; j++) {
    mutiplier[j] = (double)(rand() % 1000) / 300.0f;
  }
}

void calc_non(void) {
  for(int j = 0; j < row; j++) {
    double sum = 0;
    double *vec = mat[j];

    for(int i = 0; i < col; i++)
      sum += vec[i] * mutiplier[i];

    result[j] = sum;
  }
}

void calc_avx2(void) {
  #define VLEN (256/8/sizeof(double))

  const int col_aligned = col - col % VLEN;
  double *f_arr;

  __m256d op0, op1;
  volatile __m256d tgt;

  for (int i = 0; i < row; i++) {
    double sum = 0;

    tgt = _mm256_setzero_pd();
    for (int j = 0; j < col_aligned; j += VLEN) {
      op0 = _mm256_load_pd(&mutiplier[j]);
      op1 = _mm256_load_pd(&mat[i][j]);
      tgt = _mm256_fmadd_pd(op0, op1, tgt);
    }

    f_arr = (double *)&tgt;
    for (int k = 0; k < VLEN; k++)
      sum += f_arr[k];

    for (int l = col_aligned; l < col; l++)
      sum += mat[i][l] * mutiplier[l];

    result[i] = sum;
  }
  #undef VLEN
}

void calc_avx512(void) {
  #define VLEN (512/8/sizeof(double))
  const int col_aligned = col - col % VLEN;
  double *f_arr;

  __m512d op0, op1;
  volatile __m512d tgt;

  for (int i = 0; i < row; i++) {
    double sum = 0;

    tgt = _mm512_setzero_pd();
    for (int j = 0; j < col_aligned; j += VLEN) {
      op0 = _mm512_load_pd(&mutiplier[j]);
      op1 = _mm512_load_pd(&mat[i][j]);
      tgt = _mm512_fmadd_pd(op0, op1, tgt);
    }

    f_arr =  (double *)&tgt;
    for (int k = 0; k < VLEN; k++)
      sum += f_arr[k];

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
