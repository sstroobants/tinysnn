#include "Network.h"
#include "Connection.h"
#include "Neuron.h"
#include "Encode.h"
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

  // Initialize type as LIF
  net.type = 1;

  // Initialize output variables;
  net.out = 0.0f;
  net.tau_out = 0.9f;

  // Allocate memory for input placeholders and underlying
  // neurons and connections
//   net.in = calloc(in_size, sizeof(*net.in));
  net.enc = malloc(sizeof(*net.enc));
  net.inhid = malloc(sizeof(*net.inhid));
  net.hid = malloc(sizeof(*net.hid));
  net.hidout = malloc(sizeof(*net.hidout));

  // Call build functions for underlying neurons and connections
  *net.enc = build_encoding(hid_size);
  *net.inhid = build_connection(in_size, hid_size);
  *net.hid = build_neuron(hid_size);
  *net.hidout = build_connection(hid_size, out_size);

  return net;
}

// Init network: calls init functions for children
void init_network(Network *net) {
  // Loop over input placeholders
//   for (int i = 0; i < net->in_size; i++) {
//     net->in[i] = 0.0f;
//   }
  // Call init functions for children
  init_connection(net->inhid);
  init_neuron(net->hid);
  init_connection(net->hidout);
}

// Reset network: calls reset functions for children
void reset_network(Network *net) {
  net->out = 0.0f;
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
  // Set type
  net->type = conf->type;
  // Encoding
  load_encoding_from_header(net->enc, conf->enc);
  // Connection input -> hidden
  load_connection_from_header(net->inhid, conf->inhid);
  // Hidden neuron
  load_neuron_from_header(net->hid, conf->hid);
  // Connection hidden -> output
  load_connection_from_header(net->hidout, conf->hidout);
  // store output decay
  net->tau_out = conf->tau_out;
}


// Forward network and call forward functions for children
// Encoding and decoding inside
// TODO: but we still need to check the size of the array we put in net->in
float forward_network(Network *net) {
  // Call forward functions for children
  forward_encode(net->enc);
  // If InputALIF, process encoded values differently
  if (net->type == 2) {
    // Put encoded values in input of neuron-layer
    for (int i = 0; i < net->hid_size; i++) {
        // If it is an input alif, calculate threshold adaptation
        // bereken som van positief/negatief voor threshold update
        if (i % 2 == 0) {
            net->hid->t_s[i] = net->hid->t_s[i] - net->enc->out[i] + net->enc->out[i + 1];
            net->hid->x[i] = net->enc->out[i] * net->inhid->w[i] + net->enc->out[i + 1] * net->inhid->w[i];
        } else {
            net->hid->t_s[i] = net->hid->t_s[i] - net->enc->out[i] + net->enc->out[i - 1];
            net->hid->x[i] = net->enc->out[i] * net->inhid->w[i] + net->enc->out[i - 1] * net->inhid->w[i];
        }
        // Clamp t_s to max and min range so minimal thresh = 0, max = 2*add_thresh
        // float min_b = net->hid->th_base[i] / net->hid->add_thresh[i];
        net->hid->t_s[i] = net->hid->t_s[i] < -net->hid->th_bound[i] ? -net->hid->th_bound[i] : net->hid->t_s[i];
        net->hid->t_s[i] = net->hid->t_s[i] > net->hid->th_bound[i] ? net->hid->th_bound[i] : net->hid->t_s[i];
    }
    // Perform threshold update step
    update_thresholds(net->hid);
  } else {
      // Put encoded values in input of neuron-layer
    for (int i = 0; i < net->hid_size; i++) {
        net->hid->x[i] = net->enc->out[i] > 0 ? net->inhid->w[i] : 0.0f;
    }
  }
  forward_neuron(net->hid);
  float out_spikes = 0.0f;
  forward_connection(net->hidout, &out_spikes, net->hid->s);
  net->out = net->out * net->tau_out + out_spikes * (1 - net->tau_out);
  return net->out;
}


// Print network parameters (for debugging purposes)
void print_network(Network const *net) {
  // Input layer
//   printf("Input layer (raw):\n");
//   print_array_1d(net->in_size, net->in);
  printf("Input layer (encoded):\n");

  // Connection input -> hidden
  printf("Connection weights input -> hidden:\n");
  print_array_2d(net->hid_size, net->in_size, net->inhid->w);

  // Hidden layer
  print_neuron(net->hid);

  // Connection hidden -> output
  printf("Connection weights hidden -> output:\n");
  print_array_2d(net->out_size, net->hid_size, net->hidout->w);
}


// Free allocated memory for network and call free functions for children
void free_network(Network *net) {
  // Call free functions for children
  // Freeing in a bottom-up manner
  // TODO: or should we call this before freeing the network struct members?
  free_encode(net->enc);
  free_connection(net->inhid);
  free_neuron(net->hid);
  free_connection(net->hidout);
  // calloc() was used for input placeholders and underlying neurons and
  // connections
  free(net->enc);
  free(net->inhid);
  free(net->hid);
  free(net->hidout);
}