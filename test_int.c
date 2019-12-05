/*
 * Copyright (c) 2019, Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * * Neither the name of Intel Corporation nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * */

#include <stdio.h>
#include <x86intrin.h>
#define col 64

int vector[col];
int mutiplier[col];
int result;

void init(void) {
  for (int j = 0; j < col; j++) {
    vector[j] = j;
  }

  for (int j = 0; j < col; j++) {
    mutiplier[j] = j;
  }
}

void calc_non(void) {
  int sum = 0;

  for(int i = 0; i < col; i++)
    sum += vector[i] * mutiplier[i];
  result = sum;
}

void calc_avx512(void) {
  #define VLEN (512/8/sizeof(int))
  int *dst_arr;
  __m512i op0, op1, tmp_vec;
  volatile __m512i  tgt;
  int sum = 0;

  tgt = _mm512_setzero_si512();
  for (int j = 0; j < col; j += VLEN) {
    op0 = _mm512_load_si512(&mutiplier[j]);
    op1 = _mm512_load_si512(&vector[j]);
    tmp_vec = _mm512_mullo_epi32(op0, op1);
    tgt = _mm512_add_epi32(tmp_vec, tgt);
  }

  dst_arr = (int *)&tgt;
  for (int k = 0; k < VLEN; k++)
    sum += dst_arr[k];
  result = sum;
  #undef VLEN
}

int main() {
  init();
  calc_non();
  printf("result1 = %d\n", result);
  calc_avx512();
  printf("result2 = %d\n", result);
}
