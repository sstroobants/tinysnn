#include "Connection.h"

#include <stdio.h>
#include <stdlib.h>
// #include <./CBLAS/include/cblas.h>
// #include <arm_math.h>

// Build connection
Connection build_connection(int const pre, int const post) {
  // Connection struct
  Connection c;

  // Set shape
  c.post = post;
  c.pre = pre;

  // Allocate memory for weight array
  c.w = calloc(post * pre, sizeof(*c.w));
  c.W = (arm_matrix_instance_f32) {
    .numRows = post,
    .numCols = pre,
    .pData = c.w
  };

  return c;
}

// Init connection
void init_connection(Connection *c) {
  // Loop over weights
  for (int i = 0; i < c->post; i++) {
    for (int j = 0; j < c->pre; j++) {
      c->w[i * c->pre + j] = rand() / (float)RAND_MAX;
    }
  }
  arm_mat_init_f32(&c->W, c->post, c->pre, c->w);
}

// Reset connection
// Doesn't actually do anything, just for consistency
void reset_connection(Connection *c) {}

// Load parameters for connection (weights) from a header file
// (using the ConnectionConf struct)
void load_connection_from_header(Connection *c, ConnectionConf const *conf) {
  // Check if same shape
  if ((c->pre != conf->pre) || (c->post != conf->post)) {
    printf("Connection has a different shape than specified in the "
           "ConnectionConf!\n");
    exit(1);
  }
  // Loop over weights
  // TODO: could also be done by just exchanging pointers to arrays?
  for (int i = 0; i < c->post; i++) {
    for (int j = 0; j < c->pre; j++) {
      c->w[i * c->pre + j] = conf->w[i * c->pre + j];
    }
  }
  arm_mat_init_f32(&c->W, c->post, c->pre, c->w);
}

// Free allocated memory for connection
void free_connection(Connection *c) {
  // calloc() was used for weight array
  // Only one call, so only one free (as opposed to other methods for 2D arrays)
  free(c->w);
}

// Forward
// Spikes as floats to deal with real-valued inputs
void forward_connection(Connection *c, float x[], float const s[]) {
  // Loop over weights and multiply with spikes
  for (int i = 0; i < c->post; i++) {
    for (int j = 0; j < c->pre; j++) {
    //   printf("s[%d]: %f\n", j, s[j]);
      if (s[j] > 0.0f) {
        x[i] += c->w[i * c->pre + j];
      }
    }
  }
}

void forward_connection_real(Connection *c, float x[], float const s[]) {
  // Loop over weights and multiply with spikes
    for (int i = 0; i < c->post; i++) {
        for (int j = 0; j < c->pre; j++) {
        x[i] += c->w[i * c->pre + j] * s[j];
        }
    }
}

void forward_connection_fast(Connection *c, float x[], float const s[]) {
    arm_matrix_instance_f32 W;
    arm_matrix_instance_f32 S;
    arm_matrix_instance_f32 X;
    arm_matrix_instance_f32 X_copy;

    // Create a temporary copy of x
    float x_copy[c->post];
    memcpy(x_copy, x, c->post * sizeof(float));

    // arm_mat_init_f32(&W, c->post, c->pre, c->w);
    // arm_mat_init_f32(&S, c->pre, 1, s);
    // arm_mat_init_f32(&X, c->post, 1, x);

    arm_mat_init_f32(&W, c->post, c->pre, c->w);
    arm_mat_init_f32(&S, c->pre, 1, s);
    arm_mat_init_f32(&X, c->post, 1, x);
    arm_mat_init_f32(&X_copy, c->post, 1, x_copy);

    arm_mat_mult_f32(&W, &S, &X);
    // printf("Success: %d\n", succ);

    // add x and x_copy
    arm_mat_add_f32(&X,&X_copy, &X);
}

// void forward_connection_fast(Connection *c, float x[], float const s[]) {
//     arm_matrix_instance_q15 W;
//     arm_matrix_instance_q15 S;
//     arm_matrix_instance_q15 X;

//     // Convert weight array from float to q15
//     // q15_t q15_w[c->post * c->pre];
//     // for (int i = 0; i < c->post * c->pre; i++) {
//     //     q15_w[i] = arm_float_to_q15(c->w[i]);
//     // }

//     arm_mat_init_q15(&W, c->post, c->pre, c->w);
//     arm_mat_init_q15(&S, c->pre, 1, s);
//     arm_mat_init_q15(&X, c->post, 1, x);

//     q15_t * pState = (q15_t *)malloc(c->post * sizeof(q15_t));
//     arm_mat_mult_q15(&W, &S, &X, pState);
// }