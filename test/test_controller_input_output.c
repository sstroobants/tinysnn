#include "../NetworkController.h"
#include "../functional.h"

// Header file containing parameters
#include "../param/test_controller_conf.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Test network forward functions
int main() {
  NetworkController controller = build_network(8, 100, 83, 4, 58, 2);
  init_network(&controller);

  // Load input sequence
  int input_seq_length = 1000;
  int n_inputs = 8;
  char input_filename[] = "input.csv";
  float **inputArray = malloc(input_seq_length * sizeof(float *));
  for (int i = 0; i < input_seq_length; i++) {
    inputArray[i] = malloc(n_inputs * sizeof(float));
  }
  read_sequence(input_filename, inputArray);

  // Load network parameters from header file
  load_network_from_header(&controller, &conf);

  reset_network(&controller);

//   FILE *fptr;
//   fptr = fopen("output.csv","w");
  
  clock_t start_t, end_t;
  double total_t;

  printf("Starting loop\n");
  
  start_t = clock();
  int n_its = 50;

  for (int i_its = 0; i_its < n_its; i_its++) {
    for (int i_seq = 0; i_seq < input_seq_length; i_seq++) {
        // Set input to network from file
        set_network_input(&controller, inputArray[i_seq]);
        // print input
        // printf("Input: %f, %f, %f, %f, %f, %f, %f, %f\n", controller.in[0], controller.in[1], controller.in[2], controller.in[3], controller.in[4], controller.in[5], controller.in[6], controller.in[7]);
        // Forward network
        forward_network(&controller);
        // printf("Output: %f, %f, %f, %f\n", controller.out[0], controller.out[1], controller.integ_out[0], controller.integ_out[1]);
    }
    reset_network(&controller);
  }
  end_t = clock();
  total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
  printf("Total time taken by CPU: %f\n", total_t  );
  printf("Time per iteration: %f\n", total_t / (1000 * (float)n_its));

  // Free network memory again
  free_network(&controller);
  
  // Free input array
  for (int i = 0; i < n_inputs; i++) {
    free(inputArray[i]);
  }
  free(inputArray);
  return 0;
}