#include "../../Network.h"

// Include child structs
#include "test_deriv_enc_file.h"
#include "test_deriv_hidout_file.h"
#include "test_deriv_inhid_file.h"
#include "test_deriv_hid_file.h"
#include "test_deriv_li_out_file.h"


// in_size, hid_size, out_size, type
// inhid, hid, hidout, out
NetworkConf const conf_deriv = {1, 160, 1, 1, &conf_deriv_enc, &conf_deriv_inhid, &conf_deriv_hid, &conf_deriv_hidout, conf_deriv_li_out};
