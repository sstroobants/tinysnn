#include "../pid.h"
#include "../functional.h"

// Header file containing parameters
#include "../param/test_pid_conf.h"

#include <stdio.h>
#include <time.h>

// Test network forward functions
int main() {
  PID pid = build_pid(2, 80, 1);
  init_pid(&pid);

  // Load input sequence
  int input_seq_length = 1000;
  char input_filename[] = "inputs.csv";
  float inputArray[input_seq_length];
  read_sequence(input_filename, &inputArray, input_seq_length);

  // Load network parameters from header file
  load_pid_from_header(&pid, &conf);

  reset_pid(&pid);

  FILE *fptr;
  fptr = fopen("output.csv","w");
  
  clock_t start_t, end_t;
  double total_t;

  printf("Starting loop\n");
  
  start_t = clock();
  float output = 0.0f;
  int n_its = 500;

  for (int i_its = 0; i_its < n_its; i_its++) {
    for (int i_in = 0; i_in < input_seq_length; i_in++) {
        // Set input to network from file
        pid.in[0] = inputArray[i_in];
        pid.in[1] = inputArray[i_in];
        // Forward network
        float output_new = forward_pid(&pid);
    }
  }
  end_t = clock();
  total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
  printf("Total time taken by CPU: %f\n", total_t  );
  printf("Time per iteration: %f\n", total_t / (1000 * (float)n_its));

  // Free network memory again
  free_pid(&pid);

  return 0;
}