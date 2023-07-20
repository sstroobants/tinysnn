#pragma once

// Struct that defines an encoding layer
typedef struct Encode {
  // Encoding layer size
  int size;
  // Input
  float in;
  // Outputs
  float *out;
  // Encoding parameters
  float alpha, beta; 
} Encode;

// Struct that holds the configuration of an encoding layer
// To be used when loading parameters from a header file
typedef struct EncodeConf {
  // Encoding layer size
  int const size;
  // Encoding parameters
  float const alpha, beta;
} EncodeConf;

// Build neuron
Encode build_encoding(int const size);

// Load parameters for encoding from header file (using the EncodeConf struct)
void load_encoding_from_header(Encode *e, EncodeConf const *conf);

// Forward
void forward_encode(Encode *e);

// Free allocated memory for neuron
void free_encode(Encode *e);

// Print encoding parameters
void print_encode(Encode const *e);

