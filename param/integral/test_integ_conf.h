#include "../../Network.h"

// Include child structs
#include "test_integ_enc_file.h"
#include "test_integ_hidout_file.h"
#include "test_integ_inhid_file.h"
#include "test_integ_hid_file.h"
#include "test_integ_li_out_file.h"


// in_size, hid_size, out_size, type
// inhid, hid, hidout, out
NetworkConf const conf_integ = {1, 80, 1, 1, &conf_integ_enc, &conf_integ_inhid, &conf_integ_hid, &conf_integ_hidout, conf_integ_li_out};
