#include "../pid.h"

// Include child structs
#include "proportional/test_prop_conf.h"
#include "integral/test_integ_conf.h"
#include "derivative/test_deriv_conf.h"


// p_gain, i_gain, d_gain
// prop, integ, deriv confs
PIDConf const conf = {0.89321f, 0.79808f, 1.84817f, &conf_prop, &conf_integ, &conf_deriv};
