#pragma once

#include "Connection.h"
#include "Neuron.h"

// Struct that defines a network of two spiking layers
typedef struct NetworkController {
  // Input, encoded input, hidden and output layer sizes
  int in_size, hid_size, out_size;
  // Type (1: LIF, 2: InputALIF, ...)
  int type;
  // placeholder for input
  float *in;
  // placeholder for output and output decay
  float *out;
  float tau_out;
  // Encoding input -> hidden layer
  Connection *inhid;
  // Recurrent connection hidden -> hidden
  Connection *hidhid;
  // Hidden neurons
  Neuron *hid;
  Connection *hidout;
} NetworkController;

// Struct that holds the configuration of a two-layer network
// To be used when loading parameters from a header file
typedef struct NetworkControllerConf {
  // Input, encoded input, hidden and output layer sizes
  int const in_size, hid_size, out_size;
  // Type
  int const type;
  // Encoding input -> encoding layer
  ConnectionConf const *inhid;
  // Recurrent connection hidden -> hidden
  ConnectionConf const *hidhid;
  // Hidden neurons
  NeuronConf const *hid;
  // Connection hidden -> output
  ConnectionConf const *hidout;
  // Output decay
  const float tau_out;
} NetworkControllerConf;

// Build network: calls build functions for children
NetworkController build_network(int const in_size, int const hid_size, int const out_size);

// Init network: calls init functions for children
void init_network(NetworkController *net);

// Reset network: calls reset functions for children
void reset_network(NetworkController *net);

// Load parameters for network from header file and call load functions for
// children
void load_network_from_header(NetworkController *net, NetworkControllerConf const *conf);

// Free allocated memory for network and call free functions for children
void free_network(NetworkController *net);

// Print network parameters (for debugging purposes)
void print_network(NetworkController const *net);

// Set the inputs of the encoding layer
void set_network_input(NetworkController *net, float inputs[]);

// Forward network and call forward functions for children
// Encoding and decoding inside
float* forward_network(NetworkController *net);