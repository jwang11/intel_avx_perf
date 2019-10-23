all: matrix_float matrix_double matrix_int matrix_int16 matrix_int8 test_DLBoost

matrix_float: matrix_float.c
	gcc -O3 -mfma -mavx512f -o $@ $^

matrix_double: matrix_double.c
	gcc -O3 -mfma -mavx512f -o $@ $^

matrix_int: matrix_int.c
	gcc -O3 -mavx512vl -mavx512dq -o $@ $^

matrix_int16: matrix_int16.c
	gcc -O3 -mavx512vl -mavx512bw -o $@ $^

matrix_int8: matrix_int8.c
	gcc -O3 -mavx512vl -mavx512bw -mavx512vnni -o $@ $^

test_DLBoost: test_DLBoost.c
	g++ -O1 -mavx512vl -mavx512bw -mavx512vnni -o $@ $^

.PHONY: clean
clean:
	@rm -f matrix_float matrix_double matrix_int matrix_int16 matrix_int8 test_DLBoost
