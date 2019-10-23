/*
Copyright (c) 2018, Intel Corporation

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

#include <immintrin.h>
#include <iostream>

using namespace std;
// This code sample performs dot-product operations on 8-bit operands.
// The dot-product operation is performed first using single fused instruction,
// followed by the operation using a sequence of 3 instructions.
 
int main() {

   int8_t  op1_int8[64];
   int8_t  op2_int8[64];
   int     op3_int[16];
   int16_t op4_int16[32];

   __m512i v1_int8;
   __m512i v2_int8;
   __m512i v3_int;
   __m512i v4_int16;
   __m512i result;
   int* presult;

   // Choose some sample values
   for (int i = 0; i < 64; i++)
   {
       op1_int8[i]  = 3;
       op2_int8[i]  = 4;
   }
   for (int i = 0; i < 16; i++)
   {
       op3_int[i]   = 1;
   }
   for (int i = 0; i < 32; i++)
   {
       op4_int16[i] = 1;
   }

   //Load 512-bits of integer data
   v1_int8 =_mm512_load_si512(&op1_int8);
   v2_int8 =_mm512_load_si512(&op2_int8);
   v3_int =_mm512_load_si512(&op3_int);
   v4_int16 =_mm512_load_si512(&op4_int16);

   // PERFORM THE DOT PRODUCT OPERATION USING FUSED INSTRUCTION
   result = _mm512_dpbusds_epi32(v3_int,v1_int8,v2_int8);
   presult = (int*) &result;
   printf("RESULTS USING FUSED INSTRUCTION: \n ");
   for (int j = 15; j >= 0; j--)
       cout << presult[j]<<" ";
   cout << endl;
   cout << endl;

   // PERFORM THE DOT PRODUCT OPERATION USING A SEQUENCE OF 3 INSTRUCTIONS

   // Vertically multiply two 8-bit integers,
   // then horizontally add adjacent pairs of 16-bit integers

   __m512i vresult1 = _mm512_maddubs_epi16(v1_int8,v2_int8);

   // Upconvert to 32-bit and horizontally add neighbors. Multiply by 1.
   __m512i vresult2 = _mm512_madd_epi16(vresult1,v4_int16);

   // Add packed 32-bit integers
   result = _mm512_add_epi32(vresult2,v3_int);

   printf("RESULTS USING SEQUENCE OF 3 INSTRUCTIONS: \n ");
   presult = (int*) &result;
   for (int j = 15; j >= 0; j--)
       cout << presult[j]<<" ";
   cout << endl;

}
