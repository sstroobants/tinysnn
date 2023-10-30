from string import Template
import torch

from convert_pt_utils import create_from_template, create_connection_from_template, create_neuron_from_template


if __name__ == "__main__": 
    # Load network
    state_dict = torch.load(f"param/models/model_rate_controller_v4.pt")

    for name in state_dict.keys():
        print(name, state_dict[name])

    hidden_size = state_dict["enc.neuron.leak_i"].size()[0]

    ############# -- CONTROLLER -- #################################

    controller_conf_params = {
        'input_size': state_dict["enc.ff.weight"].size()[1],
        'encoding_size': state_dict["enc.ff.weight"].size()[0],
        'hidden_size': state_dict["rec.ff.weight"].size()[0],
        'output_size': state_dict["readout.weight"].size()[0],
        'type': 1,
    }
    controller_conf_template = 'param/templates/test_controller_conf.templ'
    controller_conf_out = 'param/test_controller_conf.h'

    create_from_template(controller_conf_template, controller_conf_out, controller_conf_params)

    ################### test_controller_inenc_file
    create_connection_from_template('inenc', state_dict, 'enc.ff.weight')

    ################### test_controller_enc_file
    create_neuron_from_template('enc', state_dict, 'enc.neuron')

    ################### test_controller_enchid_file
    create_connection_from_template('enchid', state_dict, 'rec.ff.weight')

    ################### test_controller_hidhid_file
    create_connection_from_template('hidhid', state_dict, 'rec.rec.weight')

    ################### test_controller_hid_file
    create_neuron_from_template('hid', state_dict, 'rec.neuron')

    ################### test_controller_hidout_file
    create_connection_from_template('hidout', state_dict, 'readout.weight')

    ################### test_controller_li_out

    li_out_params = {
        'leak': f"{state_dict['p_out.neuron.leak_v'][0].item()}",
        'type': "controller"
    }
    li_out_template = 'param/templates/test_li_out_file.templ'
    li_out_out = 'param/controller/test_controller_li_out_file.h'

    create_from_template(li_out_template, li_out_out, li_out_params)