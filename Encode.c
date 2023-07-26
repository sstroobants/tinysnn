#include "Encode.h"
#include "functional.h"
#include "mt19937ar.h"

#include <stdio.h>
#include <stdlib.h>

// Build neuron
Encode build_encoding(int const size) {
  // Neuron struct
  Encode e;

  // Set size
  e.size = size;
  e.in = 0.0f;
  // Allocate memory for arrays: inputs, outputs
  // No need for type casting
//   e.in = calloc(size, sizeof(*e.in));
  e.out = calloc(size, sizeof(*e.out));

  // Allocate memory for arrays: alpha, beta encoding params
  e.alpha = 0.0f;
  e.beta = 0.0f;

  // Set seed for random generator
  init_genrand(12423534895);

  return e;
}

// Load parameters for neuron from header file (using the NeuronConf struct)
void load_encoding_from_header(Encode *e, EncodeConf const *conf) {
  // Check shape
  if (e->size != conf->size) {
    printf("Encoding has a different shape than specified in the EncodeConf!\n");
    exit(1);
  }
  e->alpha = conf->alpha;
  e->beta = conf->beta;
}

// Forward
void forward_encode(Encode *e) {
    float multiplier = 1.0f;
    // iteratively process all output neurons
    for (int i = 0; i < e->size; i++) {
        float tuned = multiplier * e->beta * e->in + e->alpha;
        // calculated "tuned" value
        // Get random value TODO: PICK THIS FROM PRECREATED LIST FOR SPEED?
        float random = genrand_real2();
        // printf("|%.3f,%.3f,%.3f|", e->in, tuned, random);
        e->out[i] = tuned > random ? 1.0f : 0.0f;
        // Invert multiplier for next value
        multiplier *= -1.0f;
    }
}

// Free allocated memory for encoding
void free_encode(Encode *e) {
  free(e->out);
}

// Print encoding parameters
void print_encode(Encode const *e) {
  // Print all elements of Encoding struct
  printf("Input:\n");
  printf("%f\n", e->in);
  printf("Output:\n");
  print_array_1d(e->size, e->out);
  printf("Alpha, beta:\n");
  printf("%f, %f\n", e->alpha, e->beta);
  printf("\n");
}