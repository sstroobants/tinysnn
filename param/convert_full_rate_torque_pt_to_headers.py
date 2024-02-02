from string import Template
import torch

from convert_pt_utils import create_from_template, create_connection_from_template, create_neuron_from_template, create_connection_from_template_with_weights, create_softreset_integrator_from_template, create_connection_from_template_with_weights


if __name__ == "__main__": 
    # Load network
    state_dict = torch.load(f"param/models/model_full_effortless_sky.pt") # first full (no integrator)

    hidden_size = state_dict["l1.neuron.leak_i"].size()[0]
    hidden2_size = state_dict["l2.neuron.leak_i"].size()[0]
    hidden3_size = state_dict["l3.neuron.leak_i"].size()[0]

    ############# -- CONTROLLER -- #################################

    controller_conf_params = {
        'input_size': state_dict["l1.synapse.weight"].size()[1],
        'encoding_size': hidden_size,
        'hidden_size': hidden2_size,
        'integ_size': 4,
        'hidden2_size': hidden3_size,
        'output_size': state_dict["p_out.synapse.weight"].size()[0],
        'type': 1,
    }

    controller_conf_template = 'param/templates/test_controller_ratetorqueinteg_conf.templ'
    controller_conf_out = 'param/test_controller_conf.h'

    create_from_template(controller_conf_template, controller_conf_out, controller_conf_params)

    ################### test_controller_inenc_file
    create_connection_from_template('inenc', state_dict, 'l1.synapse.weight')

    ################### test_controller_enc_file
    create_neuron_from_template('enc', state_dict, 'l1.neuron', sigmoid=True)

    ################### test_controller_enchid_file
    create_connection_from_template('enchid', state_dict, 'l2.synapse_ff.weight')

    ################### test_controller_hidhid_file
    create_connection_from_template('hidhid', state_dict, 'l2.synapse_rec.weight')

    ################### test_controller_hid_file
    create_neuron_from_template('hid', state_dict, 'l2.neuron', sigmoid=True)

    ################### test_controller_hidinteg_file
    # Empty weight matrix, just to test
    N = controller_conf_params['integ_size']
    M = controller_conf_params['hidden_size']
    new_weights = torch.zeros([N, M])
    create_connection_from_template_with_weights('hidinteg', new_weights)

    ################### test_controller_integ_file
    create_softreset_integrator_from_template('integ')


    ################### test_controller_hidhid2_file
    create_connection_from_template('hidhid2',  state_dict, 'l3.synapse_ff.weight')

    ################### test_controller_hid2hid2_file
    create_connection_from_template('hid2hid2', state_dict, 'l3.synapse_rec.weight')

    ################### test_controller_hid2_file
    create_neuron_from_template('hid2', state_dict, 'l3.neuron', sigmoid=True)

    ################### test_controller_hid2out_file
    create_connection_from_template('hid2out', state_dict, 'p_out.synapse.weight')

    ################### test_controller_li_out

    li_out_params = {
        # 'leak': f"{torque_state_dict['p_out.neuron.leak_v'][0].item()}",
        'leak': 0.0,
        'type': "controller"
    }
    li_out_template = 'param/templates/test_li_out_file.templ'
    li_out_out = 'param/controller/test_controller_li_out_file.h'

    create_from_template(li_out_template, li_out_out, li_out_params)