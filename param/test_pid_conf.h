#include "../pid.h"

// Include child structs
#include "proportional/test_prop_conf.h"
#include "integral/test_integ_conf.h"
#include "derivative/test_deriv_conf.h"


// p_gain, i_gain, d_gain
// prop, integ, deriv confs
PIDConf const conf = {0.59063f, 0.06948f, 4.96210f, &conf_prop, &conf_integ, &conf_deriv};
