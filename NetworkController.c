#include "NetworkController.h"
#include "Connection.h"
#include "Neuron.h"
#include "Encode.h"
#include "functional.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Build network: calls build functions for children
NetworkController build_network(int const in_size, int const enc_size, int const hid_size, int const integ_size, int const hid2_size, int const out_size) {
  // Network struct
  NetworkController net;

  // Set sizes
  net.in_size = in_size;
  net.enc_size = enc_size;
  net.hid_size = hid_size;
  net.integ_size = integ_size;
  net.hid2_size = hid2_size;
  net.out_size = out_size;

  // Initialize type as LIF
  net.type = 1;

  // Initialize output variables;
  net.tau_out = 0.9f;

  // Allocate memory for input placeholders and underlying
  // neurons and connections
  net.in = calloc(in_size, sizeof(*net.in));
  net.hid2_in = calloc(hid2_size, sizeof(*net.hid2_in)); // isnt this unnecessary?
  net.inenc = malloc(sizeof(*net.inenc));
  net.enc = malloc(sizeof(*net.enc));
  net.enchid = malloc(sizeof(*net.enchid));
  net.hidhid = malloc(sizeof(*net.hidhid));
  net.hid = malloc(sizeof(*net.hid));
  net.hidinteg = malloc(sizeof(*net.hidinteg));
  net.integ = malloc(sizeof(*net.integ));
  net.hidhid2 = malloc(sizeof(*net.hidhid2));
  net.hid2hid2 = malloc(sizeof(*net.hid2hid2));
  net.hid2 = malloc(sizeof(*net.hid2));
  net.hid2out = malloc(sizeof(*net.hid2out));
  net.out = calloc(out_size, sizeof(*net.out));
  net.integ_out = calloc(2, sizeof(*net.integ_out));

  // Call build functions for underlying neurons and connections
  *net.inenc = build_connection(in_size, enc_size);
  *net.enc = build_neuron(enc_size);
  *net.enchid = build_connection(enc_size, hid_size);
  *net.hidhid = build_connection(hid_size, hid_size);
  *net.hid = build_neuron(hid_size);
  *net.hidinteg = build_connection(hid_size, integ_size);
  *net.integ = build_neuron(integ_size);
  *net.hidhid2 = build_connection(hid_size, hid2_size);
  *net.hid2hid2 = build_connection(hid2_size, hid2_size);
  *net.hid2 = build_neuron(hid2_size);
  *net.hid2out = build_connection(hid2_size, out_size);

  return net;
}

// Init network: calls init functions for children
void init_network(NetworkController *net) {
  // Call init functions for children
  init_connection(net->inenc);
  init_neuron(net->enc); 
  init_connection(net->enchid);
  init_connection(net->hidhid);
  init_neuron(net->hid);
  init_connection(net->hidinteg);
  init_neuron(net->integ);
  init_connection(net->hidhid2);
  init_connection(net->hid2hid2);
  init_neuron(net->hid2);
  init_connection(net->hid2out);
}

// Reset network: calls reset functions for children
void reset_network(NetworkController *net) {
  for (int i = 0; i < net->out_size; i++) {
    net->out[i] = 0.0f;
  }
  reset_connection(net->inenc);
  reset_neuron(net->enc);
  reset_connection(net->enchid);
  reset_connection(net->hidhid);
  reset_neuron(net->hid);
  reset_connection(net->hidinteg);
  reset_neuron(net->integ);
  reset_connection(net->hidhid2);
  reset_connection(net->hid2hid2);
  reset_neuron(net->hid2);
  reset_connection(net->hid2out);
}

// Load parameters for network from header file and call load functions for
// children
void load_network_from_header(NetworkController *net, NetworkControllerConf const *conf) {
  // Check shapes
  if ((net->in_size != conf->in_size) ||
      (net->enc_size != conf->enc_size) ||
      (net->hid_size != conf->hid_size) || 
      (net->out_size != conf->out_size) ||
      (net->integ_size != conf->integ_size) ||
      (net->hid2_size != conf->hid2_size)) {
    printf(
        "Network has a different shape than specified in the NetworkConf!\n");
    exit(1);
  }
  // Set type
  net->type = conf->type;

  // Connection input -> encoding
  load_connection_from_header(net->inenc, conf->inenc);
  // Encoding
  load_neuron_from_header(net->enc, conf->enc);
  // Connection encoding -> hidden
  load_connection_from_header(net->enchid, conf->enchid);
  // Connection hidden -> hidden
  load_connection_from_header(net->hidhid, conf->hidhid);
  // Hidden neuron
  load_neuron_from_header(net->hid, conf->hid);
  // Connection hidden-> integ
  load_connection_from_header(net->hidinteg, conf->hidinteg);
  // Integ neuron
  load_neuron_from_header(net->integ, conf->integ);
  // Connection hidden -> hidden 2
  load_connection_from_header(net->hidhid2, conf->hidhid2);
  // Connection hidden 2 -> hidden 2
  load_connection_from_header(net->hid2hid2, conf->hid2hid2);
  // Hidden 2 neuron
  load_neuron_from_header(net->hid2, conf->hid2);
  // Connection hidden -> output
  load_connection_from_header(net->hid2out, conf->hid2out);
  // store output decay
  net->tau_out = conf->tau_out;
}

// Set the inputs of the controller network with given floats
void set_network_input(NetworkController *net, float inputs[]) {
    net->in = inputs;
}


// Forward network and call forward functions for children
// Encoding and decoding inside
// TODO: but we still need to check the size of the array we put in net->in
float* forward_network(NetworkController *net) {
  forward_connection_real(net->inenc, net->enc->x, net->in);
  forward_neuron(net->enc);
  forward_connection(net->enchid, net->hid->x, net->enc->s);
  forward_connection(net->hidhid, net->hid->x, net->hid->s);
  forward_neuron(net->hid);
  forward_connection(net->hidinteg, net->integ->x, net->hid->s);
  forward_neuron(net->integ);
  
  float rate_spikes[net->hid_size];
  for (int i = 0; i < net->hid_size; i++) {
      rate_spikes[i] = 0.0f;
  }
  forward_connection(net->hidhid2, &rate_spikes, net->hid->s);
//   Run through torque network twice
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < net->hid2_size; j++) {
      net->hid2->x[j] = net->hid2->x[j] + rate_spikes[j];
    }
    // forward_connection(net->hidhid2, net->hid2->x, net->hid->s);
    forward_connection(net->hid2hid2, net->hid2->x, net->hid2->s);
    forward_neuron(net->hid2);
  }
  for (int i = 0; i < net->out_size; i++) {
    net->out[i] = 0.0f;
  }
  forward_connection(net->hid2out, net->out, net->hid2->s);
  net->integ_out[0] = net->integ_out[0] * 0.98 + (net->integ->s[0] - net->integ->s[2]) * (1.0 - 0.98);
  net->integ_out[1] = net->integ_out[1] * 0.98 + (net->integ->s[1] - net->integ->s[3]) * (1.0 - 0.98);
  return net->out;
}


// Print network parameters (for debugging purposes)
void print_network(NetworkController const *net) {
  // Input layer
//   printf("Input layer (raw):\n");
//   print_array_1d(net->in_size, net->in);
  printf("Input layer (encoded):\n");

  // Connection input -> hidden
  printf("Connection weights input -> encoding:\n");
  print_array_2d(net->enc_size, net->in_size, net->inenc->w);

  // Hidden layer
  print_neuron(net->hid);

  // Connection hidden -> output
  printf("Connection weights hidden -> output:\n");
  print_array_2d(net->out_size, net->hid_size, net->hid2out->w);
}


// Free allocated memory for network and call free functions for children
void free_network(NetworkController *net) {
  // Call free functions for children
  // Freeing in a bottom-up manner
  // TODO: or should we call this before freeing the network struct members?
  free_connection(net->inenc);
  free_neuron(net->enc);
  free_connection(net->enchid);
  free_connection(net->hidhid);
  free_neuron(net->hid);
  free_connection(net->hidinteg);
  free_neuron(net->integ);
  free_connection(net->hidhid2);
  free_connection(net->hid2hid2);
  free_neuron(net->hid2);
  free_connection(net->hid2out);
  // calloc() was used for input placeholders and underlying neurons and
  // connections
  free(net->inenc);
  free(net->enc);
  free(net->enchid);
  free(net->hidhid);
  free(net->hid);
  free(net->hidinteg);
  free(net->integ);
  free(net->hidhid2);
  free(net->hid2hid2);
  free(net->hid2);
  free(net->hid2out);
  free(net->in);
  free(net->hid2_in);
  free(net->integ_out);
  free(net->out);
}