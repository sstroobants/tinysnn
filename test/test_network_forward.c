#include "../Network.h"
#include "../functional.h"

// Header file containing parameters
#include "../param/test_network_conf.h"

#include <stdio.h>

// Test network forward functions
int main() {
  // Build network
  Network net = build_network(conf.in_size, conf.hid_size, conf.out_size);
  // Init network
  init_network(&net);
  // Set input to network
  for (int i = 0; i < conf.in_size; i++) {
    net.in[i] = 1.0f;
  }

  // Load network parameters from header file
  load_network_from_header(&net, &conf);
  reset_network(&net);

  // Forward network
  float output = forward_network(&net);

  printf("%.4f\n", output);
  // Free network memory again
  free_network(&net);

  return 0;
}