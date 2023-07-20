#include "pid.h"
#include "Network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Build network: calls build functions for children
PID build_pid(int const in_size, int const hid_size, int const out_size) {
  PID pid;
  
  // initialize input array
  pid.in = calloc(in_size, sizeof(*pid.in));
  pid.out = 0.0f;
  pid.p_gain = 1.0f;
  pid.i_gain = 0.1f;
  pid.d_gain = 0.1f;

  pid.prop = malloc(sizeof(*pid.prop));
  pid.integ = malloc(sizeof(*pid.integ));
  pid.deriv = malloc(sizeof(*pid.deriv));
  
  *pid.prop = build_network(1, hid_size, out_size);
  *pid.integ = build_network(1, hid_size, out_size);
  *pid.deriv = build_network(1, hid_size * 2, out_size);

  return pid;
}

// Init PID: calls init functions for children
void init_pid(PID *pid) {
  pid->in[0] = 0.0f; // error input
  pid->in[1] = 0.0f; // state input

  // Call init functions for children
  init_network(pid->prop);
  init_network(pid->integ);
  init_network(pid->deriv);
}

// Reset PID: calls reset functions for children
void reset_pid(PID *pid) {
  pid->in[0] = 0.0f;
  pid->in[1] = 0.0f;
  pid->out = 0.0f;
  reset_network(pid->prop);
  reset_network(pid->integ);
  reset_network(pid->deriv);
}

// Load parameters for network from header file and call load functions for
// children
void load_pid_from_header(PID *pid, PIDConf const *conf) {
  // set gains
  pid->p_gain = conf->p_gain;
  pid->i_gain = conf->i_gain;
  pid->d_gain = conf->d_gain;

  // proportional 
  load_network_from_header(pid->prop, conf->prop);
  // integral
  load_network_from_header(pid->integ, conf->integ);
  //derivative
  load_network_from_header(pid->deriv, conf->deriv);
}


// Forward pid and call forward functions for children
float forward_pid(PID *pid) {
  // set input values in all three networks
  pid->prop->enc->in = pid->in[0];
  pid->integ->enc->in = pid->in[0];
  pid->deriv->enc->in = pid->in[1];
  // forward all three networks
  float prop_out = forward_network(pid->prop) * pid->p_gain;
  float integ_out = forward_network(pid->integ) * pid->i_gain;
  float deriv_out = forward_network(pid->deriv) * pid->d_gain;

  // calculate pid output
  pid->out = prop_out + integ_out + deriv_out;
  return pid->out;
}


// Print network parameters (for debugging purposes)
void print_pid(PID const *pid) {
    print_network(pid->prop);
    print_network(pid->integ);
    print_network(pid->deriv);
}


// Free allocated memory for network and call free functions for children
void free_pid(PID *pid) {
  free_network(pid->prop);
  free_network(pid->integ);
  free_network(pid->deriv);
  free(pid->prop);
  free(pid->integ);
  free(pid->deriv);
  free(pid->in);
}