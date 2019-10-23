#ifndef PERF_H
#define PERF_H

#include <stdio.h>
#include <time.h>
#define LOOP_TIMES  100000

#define TIME(func) {\
	printf("%s\n", #func); \
	clock_t t1 = clock(); \
	func;\
	clock_t t2 = clock(); \
	printf("Time taken: %.2f second.\n", \
	((float)t2 - (float)t1) / CLOCKS_PER_SEC); \
}

#define LOOP(func) { \
  printf("loops=%d\n", LOOP_TIMES); \
  for (int r = 0; r < LOOP_TIMES; r++) {\
    func; \
  }\
}

#endif
