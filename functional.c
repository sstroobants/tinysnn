#include "functional.h"

#include <stdio.h>

// Print 1D array of floats (as floats)
void print_array_1d(int const size, float const *x) {
  for (int i = 0; i < size; i++) {
    printf("%.4f ", &x[i]);
  }
  printf("\n\n");
}

// Print 1D array of floats (as integers)
void print_array_1d_bool(int const size, float const *x) {
  for (int i = 0; i < size; i++) {
    printf("%d ", (int)&x[i]);
  }
  printf("\n\n");
}

// Print 2D array of floats (as floats)
void print_array_2d(int const rows, int const cols, float const **x) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%.4f ", &x[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

void read_sequence(char filename[], float *inputArray, int input_length) {
  FILE *input_file;
  input_file = fopen(filename, "r");
  if (input_file == NULL){
      printf("Error Reading File\n");
      exit(1);
  }
  for (int i = 0; i < input_length; i++){
      fscanf(input_file, "%f,", &inputArray[i] );
  }
  fclose(input_file);
}