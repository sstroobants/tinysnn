#include "../Encode.h"
#include "../functional.h"

// Header file containing parameters
#include "../param/test_encode_conf.h"

#include <stdio.h>

// Test encoding functions
int main() {
  // Build encoding
  Encode enc = build_encoding(conf.size);

  load_encoding_from_header(&enc, &conf);

  // Set inputs
  for (int i = 0; i < conf.size; i++) {
    enc.in[i] = 0.0f;
  }
  // Forward network
  forward_encode(&enc);

  print_array_1d(enc.size, enc.out);

  free_encode(&enc);

  return 0;
}