#include "../../Network.h"

// Include child structs
#include "test_prop_enc_file.h"
#include "test_prop_hidout_file.h"
#include "test_prop_inhid_file.h"
#include "test_prop_hid_file.h"
#include "test_prop_li_out_file.h"


// in_size, hid_size, out_size, type
// inhid, hid, hidout, out
NetworkConf const conf_prop = {1, 80, 1, 1, &conf_prop_enc, &conf_prop_inhid, &conf_prop_hid, &conf_prop_hidout, conf_prop_li_out};
