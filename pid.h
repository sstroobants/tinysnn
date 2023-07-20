#pragma once

#include "Network.h"

// Struct that defines a network of two spiking layers
typedef struct PID {
  // Two place holders: input array and output
  float *in;
  float out;
  float p_gain, i_gain, d_gain;
  // Proportional network
  Network *prop;
  // Integral network
  Network *integ;
  // Derivative network
  Network *deriv;
} PID;

// Struct that holds the configuration of a two-layer network
// To be used when loading parameters from a header file
typedef struct PIDConf {
  // gains
  float p_gain, i_gain, d_gain;
  // Proportional
  NetworkConf const *prop;
  // Integral
  NetworkConf const *integ;
  // Derivative
  NetworkConf const *deriv;
} PIDConf;

// Build network: calls build functions for children
PID build_pid(int const in_size, int const hid_size, int const out_size);

// Init network: calls init functions for children
void init_pid(PID *pid);

// Reset network: calls reset functions for children
void reset_pid(PID *pid);

// Load parameters for network from header file and call load functions for
// children
void load_pid_from_header(PID *pid, PIDConf const *conf);

// Free allocated memory for network and call free functions for children
void free_pid(PID *pid);

// Print network parameters (for debugging purposes)
void print_pid(PID const *pid);

// Forward network and call forward functions for children
// Encoding and decoding inside
float forward_pid(PID *pid);