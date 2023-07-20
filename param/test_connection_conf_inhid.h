#include "../Connection.h"

// Weights as const array here, use pointer in configuration struct
float const w_inhid[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

// post, pre, w
ConnectionConf const conf_inhid = {1, 6, w_inhid};
