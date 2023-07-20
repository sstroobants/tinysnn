#include "NetworkProportional.h"
#include "Connection.h"
#include "Neuron.h"
#include "functional.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Build network: calls build functions for children
Network build_network(int const in_size, int const hid_size, int const out_size) {
  // Network struct
  Network net;

  // Set sizes
  // Output size has to be 1
  if (out_size != 1) {
    printf("Network output size should be 1!\n");
    exit(1);
  }
  net.in_size = in_size;
  net.hid_size = hid_size;
  net.out_size = out_size;

  // Allocate memory for input placeholders, place cell centers and underlying
  // neurons and connections
  net.in = calloc(in_size, sizeof(*net.in));
  net.inhid = malloc(sizeof(*net.inhid));
  net.hid = malloc(sizeof(*net.hid));
  net.hidout = malloc(sizeof(*net.hidout));

  // Call build functions for underlying neurons and connections
  *net.inhid = build_connection(hid_size, in_size);
  *net.hid = build_neuron(hid_size);
  *net.hidout = build_connection(out_size, hid_size);
  return net;
}

// Init network: calls init functions for children
void init_network(Network *net) {
  // Loop over input placeholders
  for (int i = 0; i < net->in_size; i++) {
    net->in[i] = 0.0f;
  }
  // Call init functions for children
  init_connection(net->inhid);
  init_neuron(net->hid);
  init_connection(net->hidout);
}

// Reset network: calls reset functions for children
void reset_network(Network *net) {
  reset_connection(net->inhid);
  reset_neuron(net->hid);
  reset_connection(net->hidout);
}

// Load parameters for network from header file and call load functions for
// children
void load_network_from_header(Network *net, NetworkConf const *conf) {
  // Check shapes
  if ((net->in_size != conf->in_size) ||
      (net->hid_size != conf->hid_size) || (net->out_size != conf->out_size)) {
    printf(
        "Network has a different shape than specified in the NetworkConf!\n");
    exit(1);
  }
  // Connection input -> hidden
  load_connection_from_header(net->inhid, conf->inhid);
  // Hidden neuron
  load_neuron_from_header(net->hid, conf->hid);
  // Connection hidden -> output
  load_connection_from_header(net->hidout, conf->hidout);
}

// Free allocated memory for network and call free functions for children
void free_network(Network *net) {
  // Call free functions for children
  // Freeing in a bottom-up manner
  // TODO: or should we call this before freeing the network struct members?
  free_connection(net->inhid);
  free_neuron(net->hid);
  free_connection(net->hidout);
  // calloc() was used for input placeholders and underlying neurons and
  // connections
  free(net->in);
  free(net->inhid);
  free(net->hid);
  free(net->hidout);
}

// Print network parameters (for debugging purposes)
void print_network(Network const *net) {
  // Input layer
  printf("Input layer (raw):\n");
  print_array_1d(net->in_size, net->in);
  // Connection input -> hidden
  printf("Connection weights input -> hidden:\n");
  print_array_2d(net->hid_size, net->in_size, net->inhid->w);

  // Hidden layer
  print_neuron(net->hid);

  // Connection hidden -> output
  printf("Connection weights hidden -> output:\n");
  print_array_2d(net->out_size, net->hid_size, net->hidout->w);
}

// Forward network and call forward functions for children
// Encoding and decoding inside
// TODO: but we still need to check the size of the array we put in net->in
float forward_network(Network *net) {
  // Call forward functions for children
  forward_connection(net->inhid, net->hid->x, net->in);
  forward_neuron(net->hid);
  forward_connection(net->hidout, net->out->x, net->hid->s);
  forward_neuron(net->out);
  // Decode output neuron traces to scalar value
  float output =
      decode_network(net->out_size, net->out->t, net->decoding_scale);

  return output;
}
