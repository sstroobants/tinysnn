#include "../Connection.h"

// Weights as const array here, use pointer in configuration struct
float const w_hidout[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

// post, pre, w
ConnectionConf const conf_hidout = {6, 1, w_hidout};
