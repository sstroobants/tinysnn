#include "../Network.h"
#include "../functional.h"

// Header file containing parameters
#include "../param/proportional/test_proportional_conf.h"

#include <stdio.h>
#include <time.h>

// Test network forward functions
int main() {
  // Build network
  Network net = build_network(conf.in_size, conf.hid_size, conf.out_size);
  // Init network
  init_network(&net);

  // Load input sequence
  int input_seq_length = 1000;
  char input_filename[] = "inputs.csv";
  float inputArray[input_seq_length];
  read_sequence(input_filename, &inputArray, input_seq_length);

  // Load network parameters from header file
  load_network_from_header(&net, &conf);

  reset_network(&net);

  FILE *fptr;
  fptr = fopen("output.csv","w");
  
  clock_t start_t, end_t;
  double total_t;

  printf("Starting loop\n");
  
  start_t = clock();
  float output = 0.0f;

  for (int i_its = 0; i_its < 1000; i_its++) {
    for (int i_in = 0; i_in < input_seq_length; i_in++) {
        // Set input to network from file
        net.enc->in = inputArray[i_in];
        // Forward network
        float output_new = forward_network(&net);
    }
  }
  end_t = clock();
  total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
  printf("Total time taken by CPU: %f\n", total_t  );
  printf("Time per iteration: %f\n", total_t / (1000 * 1000));

  // Free network memory again
  free_network(&net);

  return 0;
}