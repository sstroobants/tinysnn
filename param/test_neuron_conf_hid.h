#include "../Neuron.h"

// Addition/decay/reset constants as const array here, use pointer in
// configuration struct
float const d_i[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
float const d_v[] = {0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f};

// size,d_i, d_v, v_rest
NeuronConf const conf_hid = {6, d_i, d_v, 0.0f};