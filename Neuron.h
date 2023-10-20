#pragma once

// Struct that defines a layer of neurons
// "Neuron" before and after {} to define both tag and typedef alias (as is most
// common)
typedef struct Neuron {
  // Neuron layer size
  int size;
  // Inputs
  float *x;
  // Currents
  float *i;
  // Cell voltage
  float *v;
  // Cell threshold
  float *th;
  // Cell base threshold  (for ALIF)
  float *th_base;
  // Cell threshold state (for ALIF)
  float *t_s;
  // Cell spikes
  float *s;
  // Constants (weight) for threshold adaptation
  float *add_thresh;
  // Precomputed bound for t_s
  float *th_bound;
  // Constants for decay of current, voltage
  float *d_i, *d_v;
  // Constants for resetting voltage
  float v_rest;
  // Counter for spikes
  int s_count;
} Neuron;

// Struct that holds the configuration of a layer of neurons
// To be used when loading parameters from a header file
typedef struct NeuronConf {
  // Neuron layer size
  int const size;
  // Constant for threshold adaptation
  // float const *add_thresh;
  // Constants for decay of voltage
  float const *d_i, *d_v;
  // Constants for resetting voltage and threshold
  float const v_rest;
} NeuronConf;

// Build neuron
Neuron build_neuron(int const size);

// Init neuron (addition/decay/reset constants, inputs, currents, voltage, spikes,
// threshold, trace)
void init_neuron(Neuron *n);

// Reset neuron (inputs, current, voltage, spikes, threshold, trace)
void reset_neuron(Neuron *n);

// Load parameters for neuron from header file (using the NeuronConf struct)
// d_i, d_v, v_rest, type
void load_neuron_from_header(Neuron *n, NeuronConf const *conf);

// Forward
void forward_neuron(Neuron *n);

// update threshold based on threshold state and base threshold
void update_thresholds(Neuron *n);

// Free allocated memory for neuron
void free_neuron(Neuron *n);

// Print neuron parameters
void print_neuron(Neuron const *n);

