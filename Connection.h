#pragma once

#include "arm_math.h"

// Struct that defines a connection between two layers of neurons, or input/output
typedef struct Connection {
  // Connection shape: (post, pre)
  int pre, post;
  // Connection weights
  // Use a 1D array, since structs don't support variable-sized arrays
  // TODO: is this the best way to go? Would like to be able to do w[i][j]
  //  Check this with Erik
  float *w;
  arm_matrix_instance_f32 W;
} Connection;

// Struct that holds the configuration (weights) of a connection
// To be used when loading parameters from a header file
typedef struct ConnectionConf {
  // Connection shape: (pre, post)
  int const pre, post;
  // Connection weights (1D array)
  // TODO: or actual weight array? Might be easier to specify in conf header..
  float const *w;
} ConnectionConf;

// Build connection
Connection build_connection(int const pre, int const post);

// Init connection
void init_connection(Connection *c);

// Reset connection
// Doesn't actually do anything, just for consistency
void reset_connection(Connection *c);

// Load parameters (weights) for connection from header file
// (using the ConnectionConf struct)
void load_connection_from_header(Connection *c, ConnectionConf const *conf);

// Free allocated memory for connection
void free_connection(Connection *c);

// Forward
void forward_connection(Connection *c, float x[], float const s[]);

// Forward
// Spikes as floats to deal with real-valued inputs
void forward_connection_real(Connection *c, float x[], float const s[]);

// Forward using arm_math.h
void forward_connection_fast(Connection *c, float x[], float const s[]);
