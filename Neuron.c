#include "Neuron.h"
#include "functional.h"

#include <stdio.h>
#include <stdlib.h>

// Build neuron
Neuron build_neuron(int const size) {
  // Neuron struct
  Neuron n;

  // Set size
  n.size = size;

  // Allocate memory for arrays: inputs, current, voltage, threshold, spikes, trace
  // No need for type casting
  n.x = calloc(size, sizeof(*n.x));
  n.i = calloc(size, sizeof(*n.i));
  n.v = calloc(size, sizeof(*n.v));
  n.th = calloc(size, sizeof(*n.th));
  n.th_base = calloc(size, sizeof(*n.th_base));
  n.add_thresh = calloc(size, sizeof(*n.add_thresh));
  n.th_bound = calloc(size, sizeof(*n.th_bound));
  n.t_s = calloc(size, sizeof(*n.t_s));
  n.s = calloc(size, sizeof(*n.s));

  // Allocate memory for arrays: voltage, threshold, trace and reset constants
  // Decay constants
  n.d_i = calloc(size, sizeof(*n.d_i));
  n.d_v = calloc(size, sizeof(*n.d_v));
  // Reset constants
  n.v_rest = 0.0f;

  return n;
}

// Init neuron (addition/decay/reset constants, inputs, voltage, spikes,
// threshold)
void init_neuron(Neuron *n) {
  // Loop over neurons
  for (int i = 0; i < n->size; i++) {
    // Decay constants
    n->d_i[i] = 0.8f;
    n->d_v[i] = 0.8f;
    // Inputs
    n->x[i] = 0.0f;
    // Currents
    n->i[i] = 0.0f;
    // Voltage
    n->v[i] = n->v_rest;
    // Threshold
    n->th[i] = 1.0f;
    // Base threshold 
    n->th_base[i] = 1.0f;
    // Threshold state
    n->t_s[i] = 0.0f;
    n->add_thresh[i] = 0.001f;
    n->th_bound[i] = n->th_base[i] / n->add_thresh[i];
    // Spikes
    n->s[i] = 0.0f;
  }
  // Spike counter
  n->s_count = 0;
}

// Reset neuron (inputs, voltage, spikes, threshold, trace)
void reset_neuron(Neuron *n) {
  // Loop over neurons
  for (int i = 0; i < n->size; i++) {
    // Inputs
    n->x[i] = 0.0f;
    // Currents
    n->i[i] = 0.0f;
    // Voltage
    n->v[i] = n->v_rest;
    // Threshold state
    n->t_s[i] = 0.0f;
    // Spikes
    n->s[i] = 0.0f;
  }
  // Spike counter
  n->s_count = 0;
}

// Load parameters for neuron from header file (using the NeuronConf struct)
void load_neuron_from_header(Neuron *n, NeuronConf const *conf) {
  // Check shape
  if (n->size != conf->size) {
    printf("Neuron has a different shape than specified in the NeuronConf!\n");
    exit(1);
  }
  // Loop over neurons
  for (int i = 0; i < n->size; i++) {
    // Constants for decay of voltage, threshold and trace
    n->d_i[i] = conf->d_i[i];
    n->d_v[i] = conf->d_v[i];
    // Constant for threshold adaptation
    n->add_thresh[i] = conf->add_thresh[i];
    n->th_bound[i] = n->th_base[i] / n->add_thresh[i];
  }
  // Constant for resetting voltage
  n->v_rest = conf->v_rest;
}

// get the thresholds based on threshold state and base threshold
void update_thresholds(Neuron *n) {
    for (int i = 0; i < n->size; i++) {
        n->th[i] = n->th_base[i] + n->add_thresh[i] * n->t_s[i];
    }
}


// Forward: encompasses voltage/trace/threshold updates, spiking and refraction
void forward_neuron(Neuron *n) {
    // iteratively process all neurons
    for (int i = 0; i < n->size; i++) {
        // update current
        n->i[i] = n->i[i] * n->d_i[i] + n->x[i];
        // update voltage
        n->v[i] = (n->v[i] - n->v_rest) * n->d_v[i] + n->i[i];
        // check for spike, possibly reset membrane potential and update spike count
        // TODO: NOW ONLY SOFT RESET, MAKE CONFIGURABLE?
        if (n->v[i] >= n->th[i]) {
            n->s[i] = 1.0f;
            n->v[i] = n->v[i] - n->th[i];
            // n->s_count += 1; # doing nothing with this now, remove to prevent overflow
        } else {
            n->s[i] = 0.0f;
        }
    }
}

// Free allocated memory for neuron
void free_neuron(Neuron *n) {
  // calloc() was used for voltage/decay/reset constants, inputs, voltage,
  // threshold, spike and trace arrays
  free(n->d_i);
  free(n->d_v);
  free(n->x);
  free(n->v);
  free(n->i);
  free(n->th);
  free(n->t_s);
  free(n->th_base);
  free(n->add_thresh);
  free(n->th_bound);
  free(n->s);
}

// Print neuron parameters
void print_neuron(Neuron const *n) {
  // Print all elements of neuron struct
  printf("Input:\n");
  print_array_1d(n->size, n->x);
  printf("Current:\n");
  print_array_1d(n->size, n->i);
  printf("Voltage:\n");
  print_array_1d(n->size, n->v);
  printf("Threshold:\n");
  print_array_1d(n->size, n->th);
  printf("Threshold bounds:\n");
  print_array_1d(n->size, n->th_bound);
  printf("Spikes:\n");
  print_array_1d_bool(n->size, n->s);
  printf("Decay constants:\n");
  print_array_1d(n->size, n->d_i);
  print_array_1d(n->size, n->d_v);
  printf("Reset constants threshold:\n");
  printf("Reset constant voltage: %.4f\n\n", n->v_rest);
  printf("Spike count: %d\n", n->s_count);
  printf("\n");
}